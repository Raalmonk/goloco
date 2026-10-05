"""Reusable, bounded GOLOCO inference with explicit backend and lifecycle."""
from collections import OrderedDict
import contextlib
import gc
import hashlib
from pathlib import Path
import pickle
import sys
import threading

import numpy as np
import pandas as pd
import psutil


SOURCE_COMMIT = '470477c4be6ba0095016a2089b9bbf33d49a1ef8'


class SourceIdentityError(ValueError):
    pass


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(data)
    return digest.hexdigest()


def signature(path):
    stat = Path(path).stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def aggregate_rss():
    process = psutil.Process()
    total = process.memory_info().rss
    for child in process.children(recursive=True):
        try:
            total += child.memory_info().rss
        except psutil.Error:
            pass
    return total


def resolve_model_dir(root):
    root = Path(root).resolve()
    if (root / 'L200_models').is_dir():
        return root / 'L200_models'
    if any(root.glob('model_rd10_*.pkl')):
        return root
    matches = list(root.rglob('L200_models'))
    if len(matches) != 1:
        raise ValueError('Cannot uniquely locate L200_models inside ' + str(root))
    return matches[0]


class PreparedInput:
    """Per-request values; exact original membership and input-row ordering."""
    def __init__(self, frame, experiments):
        self.frame = frame
        self.experiments = experiments
        self.matrix = frame[experiments].to_numpy().T
        self.positions = {}
        self.fallback = False
        for index, value in enumerate(frame['feature'].tolist()):
            try:
                if bool(pd.isna(value)):
                    self.fallback = True
                    continue
                self.positions.setdefault(value, []).append(index)
            except (TypeError, ValueError):
                self.fallback = True

    def select(self, features):
        use_mask = self.fallback
        if not use_mask:
            try:
                use_mask = any(bool(pd.isna(value)) for value in features)
            except (TypeError, ValueError):
                use_mask = True
        if use_mask:
            indices = np.flatnonzero(self.frame.feature.isin(features).to_numpy())
        else:
            found = set()
            for value in features:
                found.update(self.positions.get(value, []))
            indices = np.array(sorted(found), dtype=np.intp)
        return self.matrix[:, indices], indices


def output_table(genes, categories, predictions, z_scores, averages, stds, experiments):
    result = pd.DataFrame()
    result['gene'] = genes
    result = pd.merge(result, categories, on='gene', how='left')
    result['gene_category'] = result['gene_category'].replace(np.nan, 'conditional essential')
    result['avg'] = averages
    result['std'] = stds
    for index, exp in enumerate(experiments):
        result[exp + ' (CERES Pred)'] = predictions[:, index]
        result[exp + ' (Z-Score)'] = z_scores[:, index]
    result.set_index('gene')
    return result


class HeadlessGoloco:
    """Single-caller, explicit backend; original estimators remain retained.

    Cache contents are models, immutable native parameters, and metadata.
    Input values and final predictions are never cached. Methods reject
    overlapping calls; close/invalidate cannot race an in-flight prediction.
    """
    def __init__(self, repo, model_root, backend='numpy', model_bundle=None,
                 bundle_mode='mmap', library=None, max_cache_bytes=16 * 1024 ** 3,
                 process_limit_bytes=None, chunk_size=1):
        if backend not in ['numpy', 'python_fast', 'rust']:
            raise ValueError('backend must be numpy, python_fast, or rust')
        if bundle_mode not in ['mmap', 'load']:
            raise ValueError('bundle_mode must be mmap or load')
        if model_bundle is not None and backend != 'rust':
            raise ValueError('A native model bundle requires backend="rust"')
        if chunk_size != 1:
            raise NotImplementedError('This integration exposes the unchanged per-target kernel; chunk_size must be 1')
        self.repo = Path(repo).resolve()
        self.model_dir = resolve_model_dir(model_root)
        self.backend = backend
        self.model_bundle = Path(model_bundle).resolve() if model_bundle is not None else None
        self.bundle_mode, self.library_path = bundle_mode, library
        self.max_cache_bytes = int(max_cache_bytes)
        if self.max_cache_bytes <= 0 or self.max_cache_bytes > 16 * 1024 ** 3:
            raise ValueError('max_cache_bytes must be positive and no greater than 16 GiB')
        available_limit = int(psutil.virtual_memory().available * .60)
        self.process_limit_bytes = min(int(process_limit_bytes), available_limit) if process_limit_bytes is not None else available_limit
        if self.process_limit_bytes <= 0:
            raise ValueError('process_limit_bytes must be positive')
        self.chunk_size = chunk_size
        self._lock = threading.Lock()
        self._closed = True
        self._entries = OrderedDict()
        self._identities = {}
        self._bundle = None
        self._library = None
        self._statistics = {}
        self._auxiliary_bytes = 0
        self._retained_bytes = 0
        self._metadata_bytes = 0
        self._bundle_bytes = 0
        self._counters = {name: 0 for name in ['hits', 'misses', 'evictions', 'invalidations',
                                               'model_loads', 'feature_parses', 'model_hash_reads',
                                               'native_exports', 'native_export_bytes',
                                               'requests', 'estimator_calls', 'ffi_calls', 'reopens']}
        self._initialize()

    @contextlib.contextmanager
    def _operation(self, require_open=True):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('HeadlessGoloco is single-caller; another operation is active')
        try:
            if require_open and self._closed:
                raise RuntimeError('Backend is closed; call reopen() before prediction')
            yield
        finally:
            self._lock.release()

    def _initialize(self):
        self._closed = True
        self._load_metadata()
        try:
            if self.backend == 'rust':
                from .native import RustLibrary
                self._library = RustLibrary(self.library_path)
            if self.model_bundle is not None:
                from .bundle import NativeBundle
                self._bundle = NativeBundle(self.model_bundle, mode=self.bundle_mode,
                                            expected_source_commit=SOURCE_COMMIT,
                                            expected_metadata_hashes=self.metadata_hashes,
                                            max_cache_bytes=self.max_cache_bytes)
                self._bundle_bytes = int(self._bundle.accounted_bytes)
                self._bundle_signatures = {str(path): signature(path) for path in self.model_bundle.iterdir() if path.is_file()}
            self._reserve(0)
            self._closed = False
        except Exception:
            if self._bundle is not None:
                self._bundle.close()
                self._bundle = None
            self._bundle_bytes = 0
            self._library = None
            raise

    def _load_metadata(self):
        self._metadata_paths = [self.repo / 'data/19q4_sum_stats.csv', self.repo / 'data/19q4_gene_cats.csv']
        self.metadata_hashes = {path.name: file_hash(path) for path in self._metadata_paths}
        self._metadata_signatures = {str(path): signature(path) for path in self._metadata_paths}
        self.stats = pd.read_csv(self._metadata_paths[0], index_col=0)
        self.categories = pd.read_csv(self._metadata_paths[1])
        self._statistics.clear()
        self._metadata_bytes = int(self.stats.memory_usage(index=True, deep=True).sum()
                                   + self.categories.memory_usage(index=True, deep=True).sum())

    def _check_metadata(self):
        for path in self._metadata_paths:
            current = signature(path)
            if current != self._metadata_signatures[str(path)]:
                if file_hash(path) != self.metadata_hashes[path.name]:
                    raise SourceIdentityError('Reference metadata changed; explicitly invalidate all entries: ' + str(path))
                self._metadata_signatures[str(path)] = current
        if self._bundle is not None:
            for path, expected in self._bundle_signatures.items():
                if signature(path) != expected:
                    raise SourceIdentityError('Bundle files changed while open; close and verify/rebuild the bundle')

    def _model_paths(self, gene):
        if not isinstance(gene, str) or '/' in gene or '\\' in gene or gene in ['.', '..']:
            raise ValueError('Invalid target identifier')
        return (self.model_dir / ('model_rd10_' + gene + '.pkl'),
                self.model_dir / ('feats_' + gene + '.csv'))

    def _check_identity(self, gene, paths):
        current = tuple(signature(path) for path in paths)
        previous = self._identities.get(gene)
        if previous and current != previous['signatures']:
            hashes = tuple(file_hash(path) for path in paths)
            self._counters['model_hash_reads'] += 1
            if hashes != previous['hashes']:
                raise SourceIdentityError('Model or feature source changed for %s; explicitly invalidate or rebuild the bundle' % gene)
            previous['signatures'] = current
        return current

    def _evict_one(self):
        _, entry = self._entries.popitem(last=False)
        self._retained_bytes -= entry['charge']
        self._counters['evictions'] += 1

    def _reserve(self, additional):
        fixed = self._metadata_bytes + self._bundle_bytes + self._auxiliary_bytes
        if fixed + additional > self.max_cache_bytes:
            raise MemoryError('Bundle/metadata plus requested model and adapter exceed the aggregate cache budget')
        while self._entries and fixed + self._retained_bytes + additional > self.max_cache_bytes:
            self._evict_one()
        if aggregate_rss() + additional > self.process_limit_bytes:
            while self._entries and aggregate_rss() + additional > self.process_limit_bytes:
                self._evict_one()
            gc.collect()
            if aggregate_rss() + additional > self.process_limit_bytes:
                raise MemoryError('Requested model/adapter would exceed the aggregate process-RSS limit')

    def _get_entry(self, gene):
        paths = self._model_paths(gene)
        signatures = self._check_identity(gene, paths)
        if gene in self._entries:
            self._counters['hits'] += 1
            self._entries.move_to_end(gene)
            return self._entries[gene]
        self._counters['misses'] += 1
        hashes = tuple(file_hash(path) for path in paths)
        self._counters['model_hash_reads'] += 1
        info = self._bundle.model_info(gene) if self._bundle is not None else None
        if info is not None and (hashes[0] != info['source_model_sha256'] or hashes[1] != info['source_feature_sha256']):
            raise SourceIdentityError('Trusted source files do not match the native bundle for ' + gene)
        source_charge = max(paths[0].stat().st_size * 2, 65536)
        auxiliary_charge = 4096 if gene not in self._identities else 0
        self._reserve(source_charge + auxiliary_charge + 4096)
        with open(paths[0], 'rb') as handle:
            model = pickle.load(handle)
        self._counters['model_loads'] += 1
        features = pd.read_csv(paths[1], index_col=0)[0:10]['feature'].tolist()
        self._counters['feature_parses'] += 1
        feature_charge = sys.getsizeof(features) + sum(sys.getsizeof(x) for x in features) + sys.getsizeof(gene)
        adapter_charge = 2048
        if self.backend == 'numpy':
            predictor = model
        elif self.backend == 'python_fast':
            from .native import FastForest
            predictor = FastForest(model)
        else:
            from .native import RustForest, export_size
            owned_arrays = export_size(model) if self._bundle is None else 0
            self._reserve(source_charge + feature_charge + adapter_charge + owned_arrays + auxiliary_charge)
            if info is not None:
                if (features != info['feature_list'] or int(model.n_features_) != int(info['n_features'])
                        or len(model.estimators_) != int(info['n_trees'])):
                    raise SourceIdentityError('Model dimensions/features differ from the verified bundle for ' + gene)
                arrays = self._bundle.arrays_for(gene)
            else:
                arrays = None
            predictor = RustForest(model, self._library, arrays=arrays)
            if predictor.owns_export:
                adapter_charge += predictor.packed_bytes
                self._counters['native_exports'] += 1
                self._counters['native_export_bytes'] += predictor.packed_bytes
        charge = source_charge + feature_charge + adapter_charge
        self._reserve(charge + auxiliary_charge)
        if aggregate_rss() > self.process_limit_bytes:
            raise MemoryError('Model/adapter creation exceeded the aggregate process-RSS limit')
        if tuple(signature(path) for path in paths) != signatures:
            raise SourceIdentityError('Model source changed during loading for ' + gene)
        entry = {'model': model, 'predictor': predictor, 'features': features, 'charge': charge}
        self._entries[gene] = entry
        self._retained_bytes += charge
        self._auxiliary_bytes += auxiliary_charge
        self._identities[gene] = {'signatures': signatures, 'hashes': hashes}
        return entry

    def _stats_for(self, gene):
        if gene not in self._statistics:
            average, std = self.stats.loc[gene][0], self.stats.loc[gene][1]
            if self._bundle is not None:
                info = self._bundle.model_info(gene)
                if average != info['avg'] or std != info['std']:
                    raise SourceIdentityError('Reference statistics differ from the bundle for ' + gene)
            self._statistics[gene] = average, std
        return self._statistics[gene]

    def _predict(self, frame, targets, diagnostics):
        self._check_metadata()
        genes = list(targets)
        experiments = frame.columns.tolist()
        experiments.remove('feature')
        prepared = PreparedInput(frame, experiments)
        predictions = np.zeros((len(genes), len(experiments)))
        z_scores = np.zeros_like(predictions)
        averages, stds, output_genes = [], [], []
        for index, gene in enumerate(genes):
            entry = self._get_entry(gene)
            matrix, _ = prepared.select(entry['features'])
            pred = entry['predictor'].predict(matrix)
            average, std = self._stats_for(gene)
            z = (pred - average) / std
            predictions[index] = pred
            z_scores[index] = z
            averages.append(average)
            stds.append(std)
            output_genes.append(gene)
            self._counters['estimator_calls'] += 1
            if diagnostics is not None:
                diagnostics['estimator_calls'] = diagnostics.get('estimator_calls', 0) + 1
            if self.backend == 'rust':
                metrics = entry['predictor'].last_call_metrics
                self._counters['ffi_calls'] += metrics['ffi_calls']
                if diagnostics is not None:
                    for key, value in metrics.items():
                        diagnostics[key] = diagnostics.get(key, 0) + value
            # A following cache miss may evict this target. Release this local
            # model/adapter reference before loading the next target.
            del entry, pred, matrix
        self._counters['requests'] += 1
        return output_table(output_genes, self.categories, predictions, z_scores, averages, stds, experiments)

    def predict(self, frame, targets, *, diagnostics=None):
        with self._operation():
            return self._predict(frame, targets, diagnostics)

    def predict_csv(self, input_path, output_path, targets, *, diagnostics=None):
        with self._operation():
            frame = pd.read_csv(input_path)
            result = self._predict(frame, targets, diagnostics)
            result.to_csv(output_path)
            return result

    def prepare(self, targets):
        """Explicitly load/adapt targets; callers must account for this work."""
        with self._operation():
            self._check_metadata()
            for gene in targets:
                self._get_entry(gene)
            return self._snapshot()

    def invalidate(self, gene=None):
        with self._operation():
            if gene is None:
                self._entries.clear()
                self._identities.clear()
                self._statistics.clear()
                self._auxiliary_bytes = 0
                self._retained_bytes = 0
                if self._bundle is not None:
                    self._bundle.close()
                    self._bundle = None
                self._bundle_bytes = 0
                self._initialize()
            else:
                entry = self._entries.pop(gene, None)
                if entry is not None:
                    self._retained_bytes -= entry['charge']
                if self._identities.pop(gene, None) is not None:
                    self._auxiliary_bytes -= 4096
                self._statistics.pop(gene, None)
            self._counters['invalidations'] += 1

    def _snapshot(self):
        return dict(self._counters, backend=self.backend, closed=self._closed,
                    retained_models=len(self._entries), retained_model_adapter_bytes=self._retained_bytes,
                    metadata_bytes=self._metadata_bytes, bundle_accounted_bytes=self._bundle_bytes,
                    identity_statistics_accounted_bytes=self._auxiliary_bytes,
                    aggregate_cache_accounted_bytes=self._retained_bytes + self._metadata_bytes + self._bundle_bytes + self._auxiliary_bytes,
                    max_cache_bytes=self.max_cache_bytes, process_limit_bytes=self.process_limit_bytes,
                    observed_aggregate_rss_bytes=aggregate_rss(), bundle_mode=self.bundle_mode if self.model_bundle else None,
                    model_bundle=str(self.model_bundle) if self.model_bundle else None,
                    input_values_cached=False, final_predictions_cached=False, single_caller=True)

    def snapshot(self):
        with self._operation(require_open=False):
            return self._snapshot()

    def close(self):
        with self._operation(require_open=False):
            self._entries.clear()
            self._identities.clear()
            self._statistics.clear()
            self._auxiliary_bytes = 0
            self._retained_bytes = 0
            if self._bundle is not None:
                self._bundle.close()
                self._bundle = None
            self._bundle_bytes = 0
            self._library = None
            self._closed = True

    def reopen(self):
        with self._operation(require_open=False):
            if not self._closed:
                raise RuntimeError('Backend is already open')
            self._initialize()
            self._counters['reopens'] += 1
            return self

    def __enter__(self):
        if self._closed:
            self.reopen()
        return self

    def __exit__(self, *args):
        self.close()
