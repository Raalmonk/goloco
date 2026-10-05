#!/usr/bin/env python3
"""B3 versus Rust C, matched residency, original validation and CSV boundaries."""
import argparse
import ctypes
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import psutil

from goloco_benchmark import (Memory, ModelCache, Timer, atomic_json, append_jsonl,
                              check_deadline, instrumented, predict_optimized, sha256)
from goloco_experiment import correctness
from native_candidate import FastForest


class RustLibrary:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.handle = ctypes.CDLL(str(self.path))
        self.handle.goloco_forest_abi_version.restype = ctypes.c_uint32
        assert self.handle.goloco_forest_abi_version() == 1
        self.predict = self.handle.goloco_predict_forest
        self.predict.argtypes = ([ctypes.c_void_p] * 6 + [ctypes.c_size_t] * 2
                                 + [ctypes.c_void_p] + [ctypes.c_size_t] * 2
                                 + [ctypes.c_int64] * 2 + [ctypes.c_void_p])
        self.predict.restype = ctypes.c_int32


def export_size(estimator):
    return sum(x.tree_.node_count for x in estimator.estimators_) * 5 * 8 + len(estimator.estimators_) * 8


class RustForest:
    def __init__(self, estimator, library):
        FastForest(estimator)  # same exact-class, shape and scheduling guards
        self.estimator, self.library = estimator, library
        sizes = [tree.tree_.node_count for tree in estimator.estimators_]
        self.n_nodes = sum(sizes)
        self.n_trees = len(sizes)
        self.arrays = {name: np.empty(self.n_nodes, dtype=np.int64 if name in ['left', 'right', 'feature'] else np.float64)
                       for name in ['left', 'right', 'feature', 'threshold', 'value']}
        self.arrays['roots'] = np.empty(self.n_trees, dtype=np.int64)
        offset = 0
        for index, tree in enumerate(estimator.estimators_):
            native = tree.tree_
            n = native.node_count
            sl = slice(offset, offset + n)
            left, right = native.children_left, native.children_right
            assert np.array_equal(left == -1, right == -1)
            internal = left != -1
            assert np.all((left[internal] >= 0) & (left[internal] < n))
            assert np.all((right[internal] >= 0) & (right[internal] < n))
            assert np.all((native.feature[internal] >= 0) & (native.feature[internal] < estimator.n_features_))
            self.arrays['roots'][index] = offset
            self.arrays['left'][sl] = np.where(left == -1, -1, left + offset)
            self.arrays['right'][sl] = np.where(right == -1, -1, right + offset)
            self.arrays['feature'][sl] = native.feature
            self.arrays['threshold'][sl] = native.threshold
            self.arrays['value'][sl] = native.value[:, 0, 0]
            offset += n
        self.packed_bytes = sum(array.nbytes for array in self.arrays.values())
        assert self.packed_bytes == export_size(estimator)
        for array in self.arrays.values():
            assert array.flags.c_contiguous and array.flags.aligned and array.dtype.isnative
            array.flags.writeable = False
        self.pointers = [self.arrays[name].ctypes.data for name in ['left', 'right', 'feature', 'threshold', 'value', 'roots']]

    def predict(self, matrix):
        # Include the unchanged validator, dtype conversion, NumPy allocation,
        # ctypes conversion/FFI and Rust execution in every request.
        checked = self.estimator._validate_X_predict(matrix)
        if not isinstance(checked, np.ndarray):
            raise TypeError('RustForest accepts the dense GOLOCO input path only')
        assert checked.dtype == np.float32 and checked.ndim == 2
        output = np.empty(checked.shape[0], dtype=np.float64)
        status = self.library.predict(*self.pointers, self.n_nodes, self.n_trees,
                                      checked.ctypes.data, checked.shape[0], checked.shape[1],
                                      checked.strides[0], checked.strides[1], output.ctypes.data)
        if status:
            raise ValueError('Rust forest structural/argument error ' + str(status))
        return output


class CacheView:
    def __init__(self, cache, adapters):
        self.cache, self.adapters = cache, adapters
        self.stats, self.categories = cache.stats, cache.categories
    def get_model(self, gene):
        model = self.cache.get_model(gene)
        adapter = self.adapters[gene]
        assert adapter.estimator is model
        return adapter
    def get_features(self, gene):
        return self.cache.get_features(gene)
    def get_statistics(self, gene):
        return self.cache.get_statistics(gene)


def aggregate_rss():
    root = psutil.Process()
    total = root.memory_info().rss
    for child in root.children(recursive=True):
        try:
            total += child.memory_info().rss
        except psutil.Error:
            pass
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tier-results', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--reps', type=int, default=5)
    parser.add_argument('--deadline-seconds', type=float, default=900)
    args = parser.parse_args()
    assert Path('/content').exists(), 'Compile and measure only on the dedicated Colab runtime'
    args.out.mkdir(parents=True, exist_ok=True)
    assert not (args.out / 'timings.jsonl').exists(), 'Preserve completed records'
    protocol = json.loads((args.tier_results / 'protocol.json').read_text())
    repo, model_dir = Path(protocol['repo']), Path(protocol['model_dir'])
    genes, input_path = protocol['targets'], Path(protocol['input'])
    limit, budget = protocol['aggregate_memory_limit_bytes'], min(protocol['cache_budget_bytes'], 16 * 1024 ** 3)
    deadline = time.monotonic() + args.deadline_seconds
    expected = pd.read_pickle(args.tier_results / 'original/oracle_real.pkl')
    oracle_hash = sha256(args.tier_results / 'original/oracle_real.csv')
    library = RustLibrary(args.library)
    build_timer = Timer()
    with Memory() as memory:
        start, cpu = time.perf_counter(), time.process_time()
        cache = ModelCache(repo, model_dir, genes, build_timer, deadline,
                           max_bytes=budget, process_limit_bytes=limit, eager=True)
        preload_seconds, preload_cpu = time.perf_counter() - start, time.process_time() - cpu
    assert set(cache.models) == set(genes), 'Matched-residency models do not fit the declared cache'
    atomic_json(args.out / 'preload.json', {'wall_seconds': preload_seconds, 'cpu_seconds': preload_cpu,
                'memory': memory.result(), 'cache': cache.snapshot(), 'stages_seconds': build_timer.values})
    metadata_bytes = int(cache.stats.memory_usage(index=True, deep=True).sum() + cache.categories.memory_usage(index=True, deep=True).sum())
    metadata_bytes += sum(sys.getsizeof(k) + sys.getsizeof(v) + sum(sys.getsizeof(x) for x in v)
                          for k, v in cache.features.items())
    derived_expected = sum(export_size(model) for model in cache.models.values())
    assert cache.retained_bytes + metadata_bytes + derived_expected <= budget, 'Models plus exported buffers exceed the declared cache budget'
    assert aggregate_rss() + derived_expected <= limit, 'Exported buffers would exceed aggregate RSS budget'
    start, cpu = time.perf_counter(), time.process_time()
    with Memory() as memory:
        b3 = {gene: FastForest(cache.models[gene]) for gene in genes}
        native = {}
        for gene in genes:
            check_deadline(deadline)
            native[gene] = RustForest(cache.models[gene], library)
            assert aggregate_rss() <= limit
        export_seconds, export_cpu = time.perf_counter() - start, time.process_time() - cpu
    packed_bytes = sum(item.packed_bytes for item in native.values())
    assert packed_bytes == derived_expected
    atomic_json(args.out / 'export.json', {'wall_seconds': export_seconds, 'cpu_seconds': export_cpu,
                'derived_buffer_bytes': packed_bytes, 'source_models_accounted_bytes': cache.retained_bytes,
                'metadata_accounted_bytes': metadata_bytes,
                'combined_cache_accounted_bytes': packed_bytes + cache.retained_bytes + metadata_bytes,
                'cache_budget_bytes': budget, 'aggregate_rss_limit_bytes': limit, 'memory': memory.result(),
                'all_parameter_buffers_readonly': True, 'source_objects_unmodified': True})
    views = {'B3': CacheView(cache, b3), 'C': CacheView(cache, native)}
    atomic_json(args.out / 'protocol.json', {'source_protocol_sha256': sha256(args.tier_results / 'protocol.json'),
                'input_sha256': sha256(input_path), 'targets': genes, 'experiments': protocol['experiments'],
                'library_sha256': sha256(args.library), 'seed': 20261005,
                'boundary': 'CSV parse through CSV serialization; validation, conversion, ctypes FFI, allocation and output wrapping included',
                'classification': 'matched-residency diagnostic; preload/export/compile are separate first-use costs',
                'algorithm': 'unchanged ordered tree traversal; float32 input against float64 thresholds, float64 ordered leaf accumulation then one division',
                'thread_budget': 1, 'cache_budget_bytes': budget, 'aggregate_memory_limit_bytes': limit})
    sink = args.out / 'same_output_sink.csv'
    records = []

    def run(arm, rep, measured=True):
        check_deadline(deadline)
        before_cache = cache.snapshot()
        diagnostics = {}
        with Memory() as memory:
            start, cpu = time.perf_counter(), time.process_time()
            frame = pd.read_csv(input_path)
            output = predict_optimized(frame, genes, repo, model_dir, True, views[arm], deadline, diagnostics)
            output.to_csv(sink, index=protocol.get('csv_index', True))
            wall, cpu_seconds = time.perf_counter() - start, time.process_time() - cpu
        checks = correctness(expected, output, genes, protocol['experiments'])
        checks['serialized_csv_bytes_exact'] = sha256(sink) == oracle_hash
        assert checks['serialized_csv_bytes_exact']
        assert diagnostics['estimator_calls'] == len(genes)
        assert aggregate_rss() <= limit
        after_cache = cache.snapshot()
        assert after_cache['misses'] == before_cache['misses'] and after_cache['evictions'] == before_cache['evictions']
        item = {'arm': arm, 'repetition': rep, 'measured': measured, 'wall_seconds': wall, 'cpu_seconds': cpu_seconds,
                'target_count': len(genes), 'experiment_count': len(protocol['experiments']),
                'targets_per_second': len(genes)/wall, 'values_per_second': len(genes)*len(protocol['experiments'])/wall,
                'estimator_calls': diagnostics['estimator_calls'], 'memory': memory.result(),
                'cache_before': before_cache, 'cache_after': after_cache, 'correctness': checks}
        append_jsonl(args.out / ('timings.jsonl' if measured else 'prevalidation.jsonl'), item)
        if measured:
            records.append(item)
        print(json.dumps({'arm': arm, 'repetition': rep, 'measured': measured, 'wall_seconds': wall, 'exact': True}), flush=True)

    # Before timing: all real outputs match the freshly produced original oracle.
    for arm in ['B3', 'C']:
        run(arm, -1, False)
    # Additional full-target inputs are computed independently by both backends.
    frame = pd.read_csv(input_path)
    experiments = protocol['experiments']
    changed = frame.copy()
    for exp in experiments:
        changed[exp] += .01
    edge_records = []
    for name, altered in [('changed_values', changed), ('reordered_experiments', frame[['feature'] + experiments[::-1]])]:
        exp_order = [name for name in altered.columns if name != 'feature']
        baseline = predict_optimized(altered, genes, repo, model_dir, True, views['B3'], deadline)
        result = predict_optimized(altered, genes, repo, model_dir, True, views['C'], deadline)
        edge_records.append({'case': name, 'correctness': correctness(baseline, result, genes, exp_order)})
    atomic_json(args.out / 'extra_input_validation.json', edge_records)
    initial = int(np.random.RandomState(20261005).randint(2))
    orders = []
    for rep in range(args.reps):
        order = ['B3', 'C'] if (rep + initial) % 2 == 0 else ['C', 'B3']
        orders.append(order)
        for arm in order:
            run(arm, rep)
    atomic_json(args.out / 'arm_order.json', orders)
    profiles = {}
    for arm in ['B3', 'C']:
        frame = pd.read_csv(input_path)
        result, timer, _ = instrumented(frame, genes, repo, model_dir, True, views[arm], deadline)
        profiles[arm] = {'stages_seconds': timer.values, 'correctness': correctness(expected, result, genes, experiments)}
    atomic_json(args.out / 'profiles.json', profiles)
    summary = {}
    for arm in ['B3', 'C']:
        numbers = np.array([x['wall_seconds'] for x in records if x['arm'] == arm])
        summary[arm] = {'n': len(numbers), 'median_seconds': float(np.median(numbers)),
                        'min_seconds': float(numbers.min()), 'max_seconds': float(numbers.max()),
                        'iqr_seconds': float(np.percentile(numbers,75)-np.percentile(numbers,25)), 'all_exact': True}
    summary['B3_over_C'] = summary['B3']['median_seconds'] / summary['C']['median_seconds']
    summary['export_amortization_requests'] = export_seconds / (summary['B3']['median_seconds'] - summary['C']['median_seconds']) if summary['C']['median_seconds'] < summary['B3']['median_seconds'] else None
    atomic_json(args.out / 'summary.json', summary)
    pd.DataFrame([{k:v for k,v in x.items() if k not in ['memory','correctness','cache_before','cache_after']} for x in records]).to_csv(args.out/'timings.csv',index=False)
    atomic_json(args.out / 'status.json', {'status':'complete','epoch':time.time(),'samples':len(records)})
    print('RUST_COMPLETE', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
