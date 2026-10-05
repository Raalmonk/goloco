"""Optional batched FFI over the unchanged C0 forest kernel.

The core backend remains unchanged. Every matrix passes its original estimator's
validator in target order. Cache entries and NumPy owners stay pinned until the
synchronous batch call and output construction for that batch have completed.
"""
import ctypes
import time

import numpy as np

from .backend import HeadlessGoloco, PreparedInput, output_table
from .native import NativeUnavailableError


class ForestRequest(ctypes.Structure):
    """Exact repr(C) descriptor exported by goloco_predict_batch."""
    _fields_ = [
        ('left', ctypes.c_void_p), ('right', ctypes.c_void_p),
        ('feature', ctypes.c_void_p), ('threshold', ctypes.c_void_p),
        ('value', ctypes.c_void_p), ('roots', ctypes.c_void_p),
        ('left_len', ctypes.c_size_t), ('right_len', ctypes.c_size_t),
        ('feature_len', ctypes.c_size_t), ('threshold_len', ctypes.c_size_t),
        ('value_len', ctypes.c_size_t), ('roots_len', ctypes.c_size_t),
        ('n_nodes', ctypes.c_size_t), ('n_trees', ctypes.c_size_t),
        ('input', ctypes.c_void_p), ('input_low', ctypes.c_size_t),
        ('input_high', ctypes.c_size_t), ('n_samples', ctypes.c_size_t),
        ('n_features', ctypes.c_size_t), ('sample_stride_bytes', ctypes.c_int64),
        ('feature_stride_bytes', ctypes.c_int64), ('output', ctypes.c_void_p),
        ('output_len', ctypes.c_size_t),
    ]


class BatchLibrary:
    def __init__(self, library):
        self.library = library
        try:
            size = library.handle.goloco_batch_descriptor_size
            size.argtypes, size.restype = [], ctypes.c_size_t
            if size() != ctypes.sizeof(ForestRequest):
                raise NativeUnavailableError('Native batch descriptor size differs from ctypes layout')
            self.predict = library.handle.goloco_predict_batch
            self.predict.argtypes = [ctypes.POINTER(ForestRequest), ctypes.c_size_t,
                                     ctypes.POINTER(ctypes.c_size_t)]
            self.predict.restype = ctypes.c_int32
        except AttributeError as exc:
            raise NativeUnavailableError('Requested Rust library does not provide the optional batch ABI') from exc


def predict_checked_chunk(library, forests, checked_inputs):
    """Evaluate already legacy-validated matrices; retain every owner until return.

    No additional dtype conversion or change of layout occurs here. The input
    byte bounds include signed strides. One output allocation backs all returned
    row views; descriptors and all checks are constructed inside this call.
    Returns (prediction_vectors, allocation_and_ffi_metrics).
    """
    forests, checked_inputs = list(forests), list(checked_inputs)
    count = len(forests)
    if count != len(checked_inputs) or not 1 <= count <= 4096:
        raise ValueError('A chunk requires between 1 and 4096 matched forests and matrices')
    if not isinstance(library, BatchLibrary):
        library = BatchLibrary(library)
    started = time.perf_counter()
    for forest, checked in zip(forests, checked_inputs):
        if (not isinstance(checked, np.ndarray) or checked.dtype != np.float32
                or not checked.dtype.isnative or checked.ndim != 2
                or checked.shape[0] == 0 or checked.shape[1] != forest.estimator.n_features_):
            raise ValueError('Expected the original validator\'s dense float32 matrix')
    output = np.empty(sum(matrix.shape[0] for matrix in checked_inputs), dtype=np.float64)
    predictions = []
    descriptors = (ForestRequest * count)()
    offset = 0
    names = ('left', 'right', 'feature', 'threshold', 'value', 'roots')
    # Forests own parameter arrays; checked_inputs own matrix bases; output owns
    # all prediction views. All three remain live across the ctypes call.
    for index, (forest, matrix) in enumerate(zip(forests, checked_inputs)):
        values = [forest.arrays[name] for name in names]
        pred = output[offset:offset + matrix.shape[0]]
        predictions.append(pred)
        offset += matrix.shape[0]
        low, high = np.byte_bounds(matrix)
        descriptors[index] = ForestRequest(
            *[array.ctypes.data for array in values], *[array.size for array in values],
            forest.n_nodes, forest.n_trees, matrix.ctypes.data, int(low), int(high),
            matrix.shape[0], matrix.shape[1], matrix.strides[0], matrix.strides[1],
            pred.ctypes.data, pred.size)
    prepared_seconds = time.perf_counter() - started
    failed = ctypes.c_size_t()
    started = time.perf_counter()
    status = library.predict(descriptors, count, ctypes.byref(failed))
    native_seconds = time.perf_counter() - started
    if status:
        raise ValueError('Rust batch structural/argument error %d at descriptor %d' % (status, failed.value))
    return predictions, {
        'ffi_calls': 1, 'batch_targets': count, 'descriptor_count': count,
        'descriptor_allocation_bytes': ctypes.sizeof(descriptors),
        'output_allocation_bytes': int(output.nbytes), 'tracked_numpy_allocations': 1,
        'descriptor_and_output_seconds': prepared_seconds, 'native_call_seconds': native_seconds,
    }


class _FlushRequired(MemoryError):
    """All remaining cache entries are pinned by the unfinished batch."""


class ChunkedGoloco(HeadlessGoloco):
    """Optional single-caller batching with smaller batches under cache pressure.

    Pinned entries stay in the parent's charged LRU. Eviction skips them; when no
    unpinned entry remains, the pending chunk is evaluated before retrying the
    next target. Temporary matrices/descriptors/output buffers are also charged.
    chunk_size=1 delegates directly to the unchanged core request implementation.
    """
    def __init__(self, repo, model_root, backend='rust', chunk_size=64, **kwargs):
        if backend != 'rust':
            raise ValueError('ChunkedGoloco requires backend="rust"')
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or not 1 <= chunk_size <= 4096:
            raise ValueError('chunk_size must be an integer from 1 through 4096')
        self._pinned_entry_ids = set()
        self._inflight_buffer_bytes = 0
        self._chunk_library = None
        super().__init__(repo, model_root, backend='rust', chunk_size=1, **kwargs)
        self.chunk_size = chunk_size
        self._counters.update(chunk_calls=0, chunk_targets=0, chunk_budget_flushes=0)

    def _initialize(self):
        super()._initialize()
        self._chunk_library = BatchLibrary(self._library)

    def _reserve(self, additional):
        return super()._reserve(additional + self._inflight_buffer_bytes)

    def _evict_one(self):
        for gene, entry in self._entries.items():
            if id(entry) not in self._pinned_entry_ids:
                del self._entries[gene]
                self._retained_bytes -= entry['charge']
                self._counters['evictions'] += 1
                return
        raise _FlushRequired('Pending chunk pins every remaining cache entry')

    def _snapshot(self):
        result = super()._snapshot()
        result.update(chunk_size=self.chunk_size, pinned_entries=len(self._pinned_entry_ids),
                      inflight_buffer_accounted_bytes=self._inflight_buffer_bytes)
        result['aggregate_cache_accounted_bytes'] += self._inflight_buffer_bytes
        return result

    def _prepare_chunk_target(self, prepared, gene, index):
        entry = self._get_entry(gene)
        identity = id(entry)
        already_pinned = identity in self._pinned_entry_ids
        self._pinned_entry_ids.add(identity)
        try:
            # Reserve the valid matrix's worst-case conversion before allocating
            # its selection/cast buffers. Validation below rejects wrong widths.
            samples, width = prepared.matrix.shape[0], entry['model'].n_features_
            reserved = (samples * width * (prepared.matrix.dtype.itemsize + 4)
                        + samples * 8 + ctypes.sizeof(ForestRequest) + 2048)
            self._reserve(reserved)
            matrix, indices = prepared.select(entry['features'])
            checked = entry['model']._validate_X_predict(matrix)
            if not isinstance(checked, np.ndarray):
                raise TypeError('RustForest supports the dense GOLOCO matrix path')
            if checked.dtype != np.float32 or checked.ndim != 2:
                raise ValueError('Legacy validation did not produce a two-dimensional float32 matrix')
            # Preserve the original per-target failure order before considering
            # another target. Prediction remains the unchanged checked C0 kernel.
            average, std = self._stats_for(gene)
            converted = 0 if np.shares_memory(checked, matrix) else int(checked.nbytes)
            charge = int(matrix.nbytes) + converted + checked.shape[0] * 8 + ctypes.sizeof(ForestRequest) + 2048
            if charge > reserved:
                raise MemoryError('Validated chunk buffers exceed their reserved bound')
            return {'index': index, 'gene': gene, 'entry': entry, 'matrix': matrix,
                    'checked': checked, 'average': average, 'std': std,
                    'charge': charge, 'selection_bytes': int(matrix.nbytes),
                    'conversion_bytes': converted}
        except BaseException:
            if not already_pinned:
                self._pinned_entry_ids.discard(identity)
            raise

    def _predict(self, frame, targets, diagnostics):
        if self.chunk_size == 1:
            return super()._predict(frame, targets, diagnostics)
        self._check_metadata()
        genes = list(targets)
        experiments = frame.columns.tolist()
        experiments.remove('feature')
        prepared = PreparedInput(frame, experiments)
        predictions = np.zeros((len(genes), len(experiments)))
        z_scores = np.zeros_like(predictions)
        averages, stds, output_genes, pending = [], [], [], []
        raising = any(value == 'raise' for value in np.geterr().values())
        if diagnostics is not None:
            diagnostics.setdefault('actual_chunk_sizes', [])
            diagnostics['table_numpy_allocation_bytes'] = int(predictions.nbytes + z_scores.nbytes)
        def flush():
            if not pending:
                return
            result, metrics = predict_checked_chunk(self._chunk_library,
                [item['entry']['predictor'] for item in pending], [item['checked'] for item in pending])
            for item, pred in zip(pending, result):
                index = item['index']
                predictions[index] = pred
                z_scores[index] = (pred - item['average']) / item['std']
                averages.append(item['average'])
                stds.append(item['std'])
                output_genes.append(item['gene'])
            count = len(pending)
            self._counters['estimator_calls'] += count
            self._counters['ffi_calls'] += 1
            self._counters['chunk_calls'] += 1
            self._counters['chunk_targets'] += count
            if diagnostics is not None:
                diagnostics['estimator_calls'] = diagnostics.get('estimator_calls', 0) + count
                diagnostics['actual_chunk_sizes'].append(count)
                for key, value in metrics.items():
                    diagnostics[key] = diagnostics.get(key, 0) + value
                for name, key in [('input_selection_bytes', 'selection_bytes'),
                                  ('input_conversion_bytes', 'conversion_bytes')]:
                    diagnostics[name] = diagnostics.get(name, 0) + sum(item[key] for item in pending)
                diagnostics['tracked_numpy_allocations'] += count + sum(bool(item['conversion_bytes']) for item in pending)
            pending.clear()
            self._pinned_entry_ids.clear()
            self._inflight_buffer_bytes = 0
        try:
            for index, gene in enumerate(genes):
                while True:
                    retry = False
                    try:
                        item = self._prepare_chunk_target(prepared, gene, index)
                    except MemoryError:
                        if not pending:
                            raise
                        retry = True
                    except BaseException:
                        # Any earlier valid target is evaluated first, including
                        # its z-score errors, before this later error propagates.
                        flush()
                        raise
                    # Exit the exception scope before flushing: its traceback
                    # must not retain an uncharged partially loaded next model.
                    if retry:
                        self._counters['chunk_budget_flushes'] += 1
                        flush()
                        continue
                    pending.append(item)
                    self._inflight_buffer_bytes += item['charge']
                    if diagnostics is not None:
                        diagnostics['peak_chunk_buffer_accounted_bytes'] = max(
                            diagnostics.get('peak_chunk_buffer_accounted_bytes', 0), self._inflight_buffer_bytes)
                    del item
                    break
                # NumPy's optional raising policy requires earlier arithmetic to
                # happen before a later input error; use individual batches then.
                if len(pending) >= self.chunk_size or raising:
                    flush()
            flush()
            self._counters['requests'] += 1
            return output_table(output_genes, self.categories, predictions, z_scores, averages, stds, experiments)
        finally:
            pending.clear()
            self._pinned_entry_ids.clear()
            self._inflight_buffer_bytes = 0
