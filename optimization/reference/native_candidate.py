#!/usr/bin/env python3
"""B3: serial use of unchanged sklearn Cython trees, with matched B2 residency.

This is a non-Rust scheduling/wrapper optimization. Run only on the dedicated
Colab VM after the original/B1/B2 tiers. No estimator state is modified.
"""
import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted

from goloco_benchmark import (Memory, ModelCache, Timer, atomic_json,
                              append_jsonl, check_deadline, instrumented,
                              predict_optimized, sha256)
from goloco_experiment import correctness


class FastForest:
    """Narrow adapter retaining the validated forest and its original trees."""
    def __init__(self, estimator):
        if type(estimator) is not RandomForestRegressor:
            raise TypeError('B3 accepts exact RandomForestRegressor instances only')
        check_is_fitted(estimator)
        if estimator.n_outputs_ != 1:
            raise ValueError('B3 supports single-output regressors only')
        if joblib.effective_n_jobs(estimator.n_jobs) != 1:
            raise ValueError('B3 requires the original effective single-worker policy')
        if not estimator.estimators_:
            raise ValueError('B3 requires a nonempty fitted forest')
        for tree in estimator.estimators_:
            if type(tree) is not DecisionTreeRegressor:
                raise TypeError('B3 requires exact DecisionTreeRegressor members')
            check_is_fitted(tree)
            if (tree.n_outputs_ != 1 or tree.n_features_ != estimator.n_features_
                    or tree.tree_.n_outputs != 1 or tree.tree_.n_features != estimator.n_features_
                    or tree.tree_.value.shape[1:] != (1, 1)):
                raise ValueError('B3 tree dimensions differ from the guarded forest')
        self.estimator = estimator

    def predict(self, matrix):
        # This is the original forest validation, including float32 conversion,
        # finite checking and feature-count errors. No new preprocessing.
        checked = self.estimator._validate_X_predict(matrix)
        result = np.zeros(checked.shape[0], dtype=np.float64)
        # Keep accumulation order and the original native tree leaf evaluator.
        for tree in self.estimator.estimators_:
            result += tree.tree_.predict(checked)[:, 0]
        result /= len(self.estimator.estimators_)
        return result


def convert(estimator):
    """Create an adapter referencing the original model; no arrays are copied."""
    return FastForest(estimator)


class CacheView:
    """Select a predictor without replacing any object inside ModelCache."""
    def __init__(self, shared, backend, adapters):
        self.shared, self.backend, self.adapters = shared, backend, adapters
        self.stats, self.categories = shared.stats, shared.categories

    def get_model(self, gene):
        original = self.shared.get_model(gene)
        if self.backend == 'B2':
            return original
        adapter = self.adapters[gene]
        if adapter.estimator is not original:
            raise RuntimeError('Matched-residency source identity changed')
        return adapter

    def get_features(self, gene):
        return self.shared.get_features(gene)

    def get_statistics(self, gene):
        return self.shared.get_statistics(gene)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tier-results', type=Path, required=True)
    parser.add_argument('--out', '--output', dest='out', type=Path, required=True)
    parser.add_argument('--reps', type=int, default=5)
    parser.add_argument('--deadline-seconds', type=float, default=1200)
    args = parser.parse_args()
    assert Path('/content').exists(), 'Run only in the dedicated Colab runtime'
    args.out.mkdir(parents=True, exist_ok=True)
    assert not (args.out / 'timings.jsonl').exists(), 'Preserve existing completed records'
    protocol = json.loads((args.tier_results / 'protocol.json').read_text())
    repo, model_dir = Path(protocol['repo']), Path(protocol['model_dir'])
    input_path, genes = Path(protocol['input']), protocol['targets']
    expected = pd.read_pickle(args.tier_results / 'original/oracle_real.pkl')
    oracle_csv_hash = sha256(args.tier_results / 'original/oracle_real.csv')
    deadline = time.monotonic() + args.deadline_seconds
    sink = args.out / 'same_output_sink.csv'
    records = []
    atomic_json(args.out / 'protocol.json', {
        'source_protocol_sha256': sha256(args.tier_results / 'protocol.json'),
        'input_sha256': sha256(input_path), 'targets': genes, 'experiments': protocol['experiments'],
        'seed': 20261005, 'repetitions_per_arm': args.reps,
        'boundary': 'same CSV parse, common per-request PreparedInput, same metadata/model cache, output construction and CSV write',
        'classification': 'matched-residency component/application diagnostic; preload excluded and recorded separately; not a cold-service latency',
        'B3_change': 'replace joblib/tree wrappers with serial calls to unchanged sklearn native tree_.predict',
        'native_compilation_seconds': 0, 'model_array_export_seconds': 0,
        'parameters_changed': False, 'source_models_changed': False,
        'cache_budget_bytes': protocol['cache_budget_bytes'],
        'aggregate_memory_limit_bytes': protocol['aggregate_memory_limit_bytes']})
    build_timer = Timer()
    with Memory() as memory:
        before, cpu = time.perf_counter(), time.process_time()
        cache = ModelCache(repo, model_dir, genes, build_timer, deadline,
                           max_bytes=protocol['cache_budget_bytes'],
                           process_limit_bytes=protocol['aggregate_memory_limit_bytes'], eager=True)
        build_seconds, build_cpu = time.perf_counter() - before, time.process_time() - cpu
    assert set(cache.models) == set(genes), 'Matched-residency target set does not fit the declared model cache'
    atomic_json(args.out / 'preload.json', {'wall_seconds': build_seconds, 'cpu_seconds': build_cpu,
                'stages_seconds': build_timer.values, 'memory': memory.result(), 'cache': cache.snapshot(),
                'both_arms_share_exact_model_objects': True})
    identities = {gene: id(model) for gene, model in cache.models.items()}
    before = time.perf_counter()
    adapters = {gene: convert(cache.models[gene]) for gene in genes}
    atomic_json(args.out / 'adapter_setup.json', {'guard_and_wrapper_seconds': time.perf_counter() - before,
                'model_count': len(adapters), 'tree_count': sum(len(x.estimator.estimators_) for x in adapters.values()),
                'derived_model_arrays': 0, 'compiled_components': 0,
                'extra_model_references': len(adapters), 'model_parameters_and_tree_state_written': False})
    views = {name: CacheView(cache, name, adapters) for name in ['B2', 'B3']}

    def request(arm, repetition, measured):
        check_deadline(deadline)
        diagnostics = {}
        before_cache = cache.snapshot()
        with Memory() as memory:
            before, cpu = time.perf_counter(), time.process_time()
            frame = pd.read_csv(input_path)
            output = predict_optimized(frame, genes, repo, model_dir, True,
                                       views[arm], deadline, diagnostics)
            output.to_csv(sink, index=protocol.get('csv_index', True))
            wall, cpu_seconds = time.perf_counter() - before, time.process_time() - cpu
        checks = correctness(expected, output, genes, protocol['experiments'])
        checks['serialized_csv_bytes_exact'] = sha256(sink) == oracle_csv_hash
        assert checks['serialized_csv_bytes_exact']
        assert diagnostics['estimator_calls'] == len(genes)
        assert all(id(cache.models[gene]) == identities[gene] for gene in genes)
        after_cache = cache.snapshot()
        assert after_cache['misses'] == before_cache['misses']
        assert after_cache['evictions'] == before_cache['evictions']
        record = {'arm': arm, 'repetition': repetition, 'measured': measured,
                  'target_count': len(genes), 'experiment_count': len(protocol['experiments']),
                  'wall_seconds': wall, 'cpu_seconds': cpu_seconds,
                  'targets_per_second': len(genes) / wall,
                  'values_per_second': len(genes) * len(protocol['experiments']) / wall,
                  'estimator_calls': diagnostics['estimator_calls'],
                  'memory': memory.result(), 'cache_before': before_cache,
                  'cache_after': after_cache, 'correctness': checks}
        append_jsonl(args.out / ('timings.jsonl' if measured else 'prevalidation.jsonl'), record)
        if measured:
            records.append(record)
        print(json.dumps({'arm': arm, 'measured': measured, 'repetition': repetition,
                          'wall_seconds': wall, 'exact': checks['exact_frame_equal']}), flush=True)

    # Full-output checks for both methods precede every headline sample.
    for arm in ['B2', 'B3']:
        request(arm, -1, False)
    initial = int(np.random.RandomState(20261005).randint(2))
    orders = []
    for repetition in range(args.reps):
        order = ['B2', 'B3'] if (repetition + initial) % 2 == 0 else ['B3', 'B2']
        orders.append(order)
        for arm in order:
            request(arm, repetition, True)
    atomic_json(args.out / 'arm_order.json', orders)
    # Component profiling is a separate pass after all unprofiled samples.
    profiles = {}
    for arm in ['B2', 'B3']:
        check_deadline(deadline)
        frame = pd.read_csv(input_path)
        result, timer, _ = instrumented(frame, genes, repo, model_dir, True, views[arm], deadline)
        profiles[arm] = {'stages_seconds': timer.values,
                         'correctness': correctness(expected, result, genes, protocol['experiments'])}
    atomic_json(args.out / 'profiles.json', profiles)
    summaries = {}
    for arm in ['B2', 'B3']:
        values = np.array([x['wall_seconds'] for x in records if x['arm'] == arm])
        summaries[arm] = {'n': len(values), 'median_seconds': float(np.median(values)),
                          'min_seconds': float(values.min()), 'max_seconds': float(values.max()),
                          'iqr_seconds': float(np.percentile(values, 75) - np.percentile(values, 25)),
                          'all_exact': True}
    summaries['B2_over_B3'] = summaries['B2']['median_seconds'] / summaries['B3']['median_seconds']
    atomic_json(args.out / 'summary.json', summaries)
    pd.DataFrame([{k: v for k, v in x.items() if k not in ['cache_before', 'cache_after', 'memory', 'correctness']}
                  for x in records]).to_csv(args.out / 'timings.csv', index=False)
    atomic_json(args.out / 'status.json', {'status': 'complete', 'epoch': time.time(), 'samples': len(records)})
    print('B3_COMPLETE', json.dumps(summaries), flush=True)


if __name__ == '__main__':
    main()
