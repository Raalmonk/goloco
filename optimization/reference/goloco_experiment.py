#!/usr/bin/env python3
"""Fresh-worker, unprofiled CSV-to-CSV GOLOCO experiment runner.

Run exclusively in the one dedicated Colab runtime. The Mac is an editor
and orchestrator only. See protocol.json and each worker's raw records.
"""
import argparse
from collections import Counter
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import pickle
import platform
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd
import psutil

from goloco_benchmark import (HeadlessGoloco, Memory, ModelCache, PreparedInput,
                              Timer, atomic_json, append_jsonl, check_deadline,
                              describe_model, feature_list, instrumented,
                              load_author, original_output, resolve_model_dir,
                              sha256, validate)

SEED = 20261005
ATOL, RTOL = 1e-8, 1e-7


def plan(args):
    out = args.results
    out.mkdir(parents=True, exist_ok=True)
    model_dir = resolve_model_dir(args.models)
    canonical = list(pd.read_pickle(args.repo / 'data/19q4_genes.pkl'))
    stats = pd.read_csv(args.repo / 'data/19q4_sum_stats.csv', index_col=0)
    exclusions, supported = [], []
    for gene in canonical:
        reasons = []
        if not (model_dir / ('model_rd10_' + gene + '.pkl')).is_file():
            reasons.append('missing_model')
        if not (model_dir / ('feats_' + gene + '.csv')).is_file():
            reasons.append('missing_feature_manifest')
        if gene not in stats.index:
            reasons.append('missing_statistic')
        elif not np.isfinite(stats.loc[gene].to_numpy()[:2]).all() or stats.loc[gene][1] == 0:
            reasons.append('invalid_reference_statistic')
        if reasons:
            exclusions.append({'gene': gene, 'reasons': reasons})
        else:
            supported.append(gene)
    size = len(supported) if args.size == 'full' else int(args.size)
    if size > len(supported):
        raise ValueError('Requested target count exceeds supported scope.')
    order = np.random.RandomState(SEED).permutation(len(supported))
    indices = np.arange(size) if size == 16 else np.sort(order[:size])
    genes = [supported[i] for i in indices]
    if size < len(supported):
        alternate_indices = np.sort(np.roll(order, -min(16, len(supported) - size))[:size])
        alternate = [supported[i] for i in alternate_indices]
    else:
        alternate = list(reversed(genes))
    input_path = (args.input or args.repo / 'data/PC9SKMELCOLO_L200_CERES_Features.csv').resolve()
    real = pd.read_csv(input_path)
    experiments = [x for x in real.columns if x != 'feature']
    if experiments != ['ACH-000030', 'ACH-000615', 'ACH-001042']:
        raise ValueError('Supplied input experiment columns differ from expected author example.')
    cases_dir = out / 'request_inputs'
    cases_dir.mkdir(exist_ok=True)
    changed = real.copy()
    for exp in experiments:
        changed[exp] = changed[exp] + 0.01
    changed_path = cases_dir / 'changed_numeric_input.csv'
    changed.to_csv(changed_path, index=False)
    # Input identity is raw author CSV for the real workload. No rewrite.
    cases = {'real': {'input': str(input_path), 'targets': genes, 'biological_example': True},
             'changed_values': {'input': str(changed_path), 'targets': genes, 'biological_example': False},
             'changed_targets': {'input': str(input_path), 'targets': alternate, 'biological_example': True}}
    repeats = args.repeats if args.repeats is not None else (5 if size == 1024 else 3)
    sequence = ['real'] + (['real', 'changed_values', 'changed_targets', 'real', 'real'] * (1 + repeats // 5))[:repeats]
    candidate_order = [x for x in args.variants.split(',') if x != 'original']
    np.random.RandomState(SEED + size).shuffle(candidate_order)
    arm_order = ['original'] + candidate_order
    available_ram = psutil.virtual_memory().available
    protocol = {'seed': SEED, 'canonical_count': len(canonical), 'supported_count': len(supported),
                'target_count': size, 'targets': genes, 'exclusions': exclusions,
                'selection': 'first16 canonical smoke; other numeric tiers seed20261005 permutation prefix restored to canonical order; full supported canonical order',
                'input': str(input_path), 'input_sha256': sha256(input_path),
                'input_shape': list(real.shape), 'input_dtypes': {k: str(v) for k, v in real.dtypes.items()},
                'input_missing_values': real.isna().sum().to_dict(),
                'input_feature_order': real.feature.tolist(), 'experiments': experiments,
                'cases': cases, 'request_sequence': sequence, 'arm_order': arm_order,
                'arm_order_policy': 'original first establishes fresh oracle; candidate arms seeded randomized; each arm fresh worker',
                'cache_budget_bytes': args.cache_gib * 1024 ** 3,
                'available_ram_at_plan_bytes': available_ram,
                'aggregate_memory_limit_bytes': int(available_ram * 0.60),
                'atol': ATOL, 'rtol': RTOL, 'csv_index': True,
                'headline_boundary': 'fresh CSV parse through final CSV write to same runtime-local sink; imports/provenance/prevalidation excluded; cold worker has no model/metadata store',
                'os_page_cache': 'uncontrolled; no physical-disk coldness claim',
                'repeated_distribution': 'deterministic mixed service requests; identical real-request samples also reported separately',
                'cache_policy': 'lazy bounded LRU; first request includes required metadata and model population; no preloading; source artifacts immutable; explicit invalidation API',
                'process_imports': 'outside inference wall time; parent records total worker wall time separately',
                'repo': str(args.repo.resolve()), 'model_dir': str(model_dir.resolve())}
    atomic_json(out / 'protocol.json', protocol)
    atomic_json(out / 'targets.json', genes)
    return protocol


def prevalidate(protocol, out):
    """Require exact selection and matrix equality before candidate timing."""
    model_dir = Path(protocol['model_dir'])
    manifests = {}
    for name in dict.fromkeys(protocol['request_sequence']):
        case = protocol['cases'][name]
        frame = pd.read_csv(case['input'])
        experiments = [x for x in frame.columns if x != 'feature']
        prepared = PreparedInput(frame, experiments)
        rows = {}
        for gene in case['targets']:
            features = feature_list(model_dir, gene)
            actual, indices = prepared.select(features)
            mask = frame.feature.isin(features)
            original_indices = np.flatnonzero(mask.to_numpy())
            original = frame[mask][experiments].to_numpy().T
            assert np.array_equal(indices, original_indices), gene
            assert actual.dtype == original.dtype and actual.shape == original.shape, gene
            assert np.array_equal(actual, original, equal_nan=True), gene
            assert frame.iloc[indices].feature.tolist() == frame[mask].feature.tolist(), gene
            rows[gene] = {'indices': indices.tolist(), 'feature_ids': frame.iloc[indices].feature.tolist(),
                          'dtype': str(actual.dtype), 'shape': list(actual.shape),
                          'matrix_sha256': hashlib.sha256(actual.tobytes(order='C')).hexdigest()}
        atomic_json(out / ('prevalidation_' + name + '.json'), rows)
        manifests[name] = {'targets': len(rows), 'all_exact': True,
                           'sha256': sha256(out / ('prevalidation_' + name + '.json'))}
    atomic_json(out / 'prevalidation.json', manifests)
    gc.collect()


def correctness(expected, actual, genes, experiments):
    result = validate(expected, actual, genes, experiments)
    numeric = actual.select_dtypes(include=[np.number]).to_numpy()
    reference = expected.select_dtypes(include=[np.number]).to_numpy()
    nan = np.isnan(numeric)
    posinf, neginf = np.isposinf(numeric), np.isneginf(numeric)
    assert np.array_equal(nan, np.isnan(reference))
    assert np.array_equal(posinf, np.isposinf(reference))
    assert np.array_equal(neginf, np.isneginf(reference))
    finite = np.isfinite(numeric) & np.isfinite(reference)
    absolute = np.abs(numeric[finite] - reference[finite])
    denominator = np.abs(reference[finite])
    relative = np.divide(absolute, denominator, out=np.zeros_like(absolute), where=denominator != 0)
    relative[(denominator == 0) & (absolute != 0)] = np.inf
    result.update({'declared_atol': ATOL, 'declared_rtol': RTOL,
                   'max_relative_error': float(relative.max()) if len(relative) else 0.0,
                   'nan_locations': np.argwhere(nan).tolist(),
                   'positive_inf_locations': np.argwhere(posinf).tolist(),
                   'negative_inf_locations': np.argwhere(neginf).tolist(),
                   'invalid_masks_match': True, 'within_declared_tolerances': bool(np.allclose(numeric, reference, atol=ATOL, rtol=RTOL, equal_nan=True))})
    return result


def worker(args):
    protocol = json.loads((args.results / 'protocol.json').read_text())
    arm = args.worker
    out = args.results / arm
    out.mkdir(exist_ok=True)
    if (out / 'timings.jsonl').exists():
        raise ValueError('Worker results already exist; use a new tier directory.')
    deadline = time.monotonic() + args.deadline_seconds
    logging.disable(logging.CRITICAL)
    repo, model_dir = Path(protocol['repo']), Path(protocol['model_dir'])
    # Same exact matrix prevalidation for all arms, before any measured call.
    before = time.perf_counter()
    prevalidate(protocol, out)
    prevalidation_seconds = time.perf_counter() - before
    module, original = load_author(repo, model_dir, protocol['targets'], out, neutral_progress=True)
    import sklearn
    import scipy
    import joblib
    environment = {'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__,
                   'sklearn': sklearn.__version__, 'scipy': scipy.__version__, 'joblib': joblib.__version__,
                   'pid': os.getpid(), 'platform': platform.platform(), 'logical_cpus': os.cpu_count(),
                   'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                   'thread_limits': {k: os.environ.get(k) for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']},
                   'prevalidation_seconds': prevalidation_seconds,
                   'headless_adapter': 'author infer methods and app table construction unchanged; scoped targets substituted; Dash callbacks, progress/ETA arithmetic, logging output, persistence and UUID omitted equally from allarms',
                   'memory_accounting': 'sampled aggregate parent plus descendant RSS at10ms; cache checked against explicit60%initialavailableRAM limit',
                   'cache_budget_bytes': protocol['cache_budget_bytes']}
    atomic_json(out / 'environment.json', environment)
    engine = None
    records = []
    estimator_counter = {'count': 0}
    original_infer_gene = module.infer.infer_gene
    def counted_original(self, gene):
        check_deadline(deadline)
        estimator_counter['count'] += 1
        return original_infer_gene(self, gene)
    module.infer.infer_gene = counted_original
    same_sink = args.results / 'timed_output_sink.csv'
    status = {'state': 'running', 'variant': arm}
    atomic_json(out / 'status.json', status)
    try:
        for i, case_name in enumerate(protocol['request_sequence']):
            check_deadline(deadline)
            case = protocol['cases'][case_name]
            genes = case['targets']
            # Load oracle only for checking, outside the inference boundary.
            oracle_path = args.results / 'original' / ('oracle_' + case_name + '.pkl')
            expected = pd.read_pickle(oracle_path) if oracle_path.exists() else None
            before_cache = engine.cache.snapshot() if engine is not None and engine.cache is not None else None
            estimator_counter['count'] = 0
            diagnostics = {}
            with Memory() as memory:
                cpu_before = time.process_time()
                wall_before = time.perf_counter()
                frame = pd.read_csv(case['input'])
                if arm == 'original':
                    original.__globals__['_goloco_scope'] = genes
                    output = original_output(original, frame, deadline)
                else:
                    if engine is None:
                        engine = HeadlessGoloco(repo, model_dir,
                                                max_cache_bytes=protocol['cache_budget_bytes'],
                                                process_limit_bytes=protocol['aggregate_memory_limit_bytes'],
                                                use_index=arm in ['index_only', 'combined'],
                                                use_cache=arm in ['cache_only', 'combined'])
                    output = engine.predict(frame, genes, deadline, diagnostics)
                # Author download uses pandas.to_csv default index=True.
                output.to_csv(same_sink)
                elapsed = time.perf_counter() - wall_before
                cpu_seconds = time.process_time() - cpu_before
            if memory.peak > protocol['aggregate_memory_limit_bytes']:
                raise MemoryError('Aggregate RSS exceeded experiment memory limit.')
            experiments = [x for x in frame.columns if x != 'feature']
            if expected is None:
                if arm != 'original':
                    raise ValueError('Original reference missing for ' + case_name)
                expected = output.copy(deep=True)
                expected.to_pickle(oracle_path)
                shutil.copyfile(same_sink, out / ('oracle_' + case_name + '.csv'))
            checks = correctness(expected, output, genes, experiments)
            oracle_csv = args.results / 'original' / ('oracle_' + case_name + '.csv')
            checks['serialized_csv_bytes_exact'] = sha256(same_sink) == sha256(oracle_csv)
            if not checks['serialized_csv_bytes_exact']:
                raise AssertionError('Serialized CSV differs from original.')
            call_count = estimator_counter['count'] if arm == 'original' else diagnostics['estimator_calls']
            assert call_count == len(genes)
            after_cache = engine.cache.snapshot() if engine is not None and engine.cache is not None else None
            record = {'variant': arm, 'request_index': i, 'request_case': case_name,
                      'state': 'application_cold' if i == 0 else 'repeated_service',
                      'targets': len(genes), 'experiments': len(experiments),
                      'wall_seconds': elapsed, 'cpu_seconds': cpu_seconds,
                      'targets_per_second': len(genes) / elapsed,
                      'predicted_values_per_second': len(genes) * len(experiments) / elapsed,
                      'memory': memory.result(), 'cache_before': before_cache, 'cache_after': after_cache,
                      'estimator_calls': call_count, 'correctness': checks,
                      'input_sha256': sha256(case['input']),
                      'target_ids_sha256': hashlib.sha256(json.dumps(genes).encode()).hexdigest(),
                      'csv_sha256': sha256(same_sink)}
            records.append(record)
            append_jsonl(out / 'timings.jsonl', record)
            atomic_json(out / 'last_completed_request.json', record)
            print(json.dumps({k: record[k] for k in ['variant', 'request_index', 'request_case', 'wall_seconds', 'cpu_seconds', 'targets']}), flush=True)
            if i == 0:
                shutil.copyfile(same_sink, out / 'real_prediction.csv')
        # Profile one independent request AFTER headline measurements.
        check_deadline(deadline)
        profile_frame = pd.read_csv(protocol['input'])
        profile_cache = engine.cache if engine is not None else None
        if arm == 'original':
            use_index = False
        else:
            use_index = arm in ['index_only', 'combined']
        profile_before = time.perf_counter()
        result, timer, _ = instrumented(profile_frame, protocol['targets'], repo,
                                        model_dir, use_index, profile_cache, deadline)
        profile_elapsed = time.perf_counter() - profile_before
        correctness(pd.read_pickle(args.results / 'original/oracle_real.pkl'), result,
                    protocol['targets'], protocol['experiments'])
        serialized_before = time.perf_counter()
        result.to_csv(same_sink)
        serialization = time.perf_counter() - serialized_before
        profile_record = {'profile_scope': 'one post-headline request; same retained modelstore if implemented',
                          'stages_seconds': timer.values, 'inference_profile_wall_seconds': profile_elapsed,
                          'csv_serialization_seconds': serialization,
                          'diagnostic_expected_counts': {'model_deserializations': 0 if profile_cache else len(protocol['targets']),
                                                         'feature_csv_parses': 0 if profile_cache else len(protocol['targets']),
                                                         'estimator_calls': len(protocol['targets'])},
                          'nested_stage_sum_is_not_added_to_wall': True}
        atomic_json(out / 'profile.json', profile_record)
        status = {'state': 'complete', 'variant': arm, 'completed_requests': len(records)}
    except Exception as exc:
        status = {'state': 'failed', 'variant': arm, 'completed_requests': len(records),
                  'error': repr(exc), 'traceback': traceback.format_exc()}
        raise
    finally:
        atomic_json(out / 'status.json', status)


def summarize(args):
    records = []
    for path in args.results.glob('*/timings.jsonl'):
        records.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    if not records:
        return
    with open(args.results / 'timings.jsonl', 'w') as handle:
        for row in records:
            handle.write(json.dumps(row) + '\n')
    flat, summaries = [], []
    for row in records:
        item = {k: v for k, v in row.items() if not isinstance(v, dict)}
        item.update(row['memory'])
        item['exact_output'] = row['correctness']['exact_frame_equal']
        flat.append(item)
    pd.DataFrame(flat).to_csv(args.results / 'timings.csv', index=False)
    for variant in dict.fromkeys(row['variant'] for row in records):
        rows = [r for r in records if r['variant'] == variant]
        for population in ['all_repeated_service', 'identical_real_repeated']:
            repeated = [r for r in rows if r['request_index'] > 0 and (population == 'all_repeated_service' or r['request_case'] == 'real')]
            values = [r['wall_seconds'] for r in repeated]
            cold = next((r for r in rows if r['request_index'] == 0), None)
            summaries.append({'variant': variant, 'population': population,
                              'targets': rows[0]['targets'], 'experiments': rows[0]['experiments'],
                              'application_cold_seconds': cold['wall_seconds'] if cold else None,
                              'repeated_samples': len(values),
                              'median_seconds': float(np.median(values)) if values else None,
                              'min_seconds': min(values) if values else None, 'max_seconds': max(values) if values else None,
                              'iqr_seconds': float(np.percentile(values, 75) - np.percentile(values, 25)) if values else None,
                              'peak_aggregate_rss_bytes': max(r['memory']['sampled_peak_rss_bytes'] for r in rows),
                              'all_exact': all(r['correctness']['exact_frame_equal'] and r['correctness']['serialized_csv_bytes_exact'] for r in rows)})
    pd.DataFrame(summaries).to_csv(args.results / 'summary.csv', index=False)
    atomic_json(args.results / 'summary.json', summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--size', default='16')
    parser.add_argument('--repeats', type=int)
    parser.add_argument('--variants', default='original,index_only,cache_only,combined')
    parser.add_argument('--cache-gib', type=float, default=16)
    parser.add_argument('--deadline-seconds', type=float, default=1200)
    parser.add_argument('--worker', choices=['original', 'index_only', 'cache_only', 'combined'])
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    args.repo = args.repo.resolve()
    args.models = args.models.resolve()
    args.results = args.results.resolve()
    if args.summarize_only:
        summarize(args)
        return
    if args.worker:
        worker(args)
        return
    if (args.results / 'protocol.json').exists():
        raise ValueError('Tier already planned; preserve it and use a new result directory.')
    protocol = plan(args)
    if args.plan_only:
        return
    deadline = time.monotonic() + args.deadline_seconds
    for arm in protocol['arm_order']:
        remaining = deadline - time.monotonic()
        if remaining < 10:
            break
        command = [sys.executable, str(Path(__file__).resolve()), '--repo', str(args.repo),
                   '--models', str(args.models), '--results', str(args.results),
                   '--worker', arm, '--deadline-seconds', str(max(1, remaining - 5))]
        before = time.perf_counter()
        with open(args.results / (arm + '_worker.log'), 'w') as handle:
            try:
                process = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                                         timeout=remaining, check=False)
                outcome = {'variant': arm, 'returncode': process.returncode,
                           'worker_total_wall_seconds': time.perf_counter() - before}
            except subprocess.TimeoutExpired:
                outcome = {'variant': arm, 'timeout': True,
                           'worker_total_wall_seconds': time.perf_counter() - before}
        append_jsonl(args.results / 'workers.jsonl', outcome)
        summarize(args)
        print(json.dumps(outcome), flush=True)
        if outcome.get('returncode', 1) != 0:
            break


if __name__ == '__main__':
    main()
