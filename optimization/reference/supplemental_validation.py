#!/usr/bin/env python3
"""Remote-only GOLOCO validation; no performance claims or saved-output inference.

Run in the same dedicated Colab environment as goloco_benchmark.py. Source
models are read only. All output files are created beneath --results.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import pickle
import subprocess
import time
import traceback

import numpy as np
import pandas as pd

import goloco_benchmark as bench

ATOL = 1e-8
RTOL = 1e-7
SEED = 20261005


def numeric_masks(values):
    array = np.asarray(values, dtype=np.float64)
    return {'shape': list(array.shape),
            'nan_locations': np.argwhere(np.isnan(array)).tolist(),
            'positive_inf_locations': np.argwhere(np.isposinf(array)).tolist(),
            'negative_inf_locations': np.argwhere(np.isneginf(array)).tolist(),
            'finite_count': int(np.isfinite(array).sum())}


def frame_comparison(reference, candidate):
    structural = {'columns_exact': reference.columns.tolist() == candidate.columns.tolist(),
                  'index_exact': reference.index.equals(candidate.index),
                  'index_type_exact': type(reference.index) is type(candidate.index),
                  'index_names_exact': reference.index.names == candidate.index.names,
                  'dtypes_exact': reference.dtypes.astype(str).tolist() == candidate.dtypes.astype(str).tolist(),
                  'shape_exact': reference.shape == candidate.shape}
    if not all(structural.values()):
        return dict(structural, accepted=False, exact_match=False)
    numeric_columns = reference.select_dtypes(include=[np.number]).columns.tolist()
    text_columns = [c for c in reference.columns if c not in numeric_columns]
    text_exact = reference[text_columns].equals(candidate[text_columns])
    left = reference[numeric_columns].to_numpy()
    right = candidate[numeric_columns].to_numpy()
    left_masks, right_masks = numeric_masks(left), numeric_masks(right)
    masks_exact = left_masks == right_masks
    finite = np.isfinite(left) & np.isfinite(right)
    absolute = np.abs(left[finite] - right[finite])
    denominator = np.abs(left[finite])
    relative = np.divide(absolute, denominator, out=np.zeros_like(absolute), where=denominator != 0)
    relative[(denominator == 0) & (absolute != 0)] = np.inf
    exact = bool(np.array_equal(left, right, equal_nan=True)) and text_exact
    close = masks_exact and bool(np.allclose(left[finite], right[finite], atol=ATOL, rtol=RTOL))
    return dict(structural, nonnumeric_exact=text_exact, numeric_columns=numeric_columns,
                reference_special_values=left_masks, candidate_special_values=right_masks,
                special_value_masks_exact=masks_exact, exact_match=exact,
                tolerance_atol=ATOL, tolerance_rtol=RTOL,
                max_absolute_error=float(absolute.max()) if absolute.size else 0.0,
                max_relative_error=float(relative.max()) if relative.size else 0.0,
                accepted=bool(close and text_exact and all(structural.values())))


def upload_gate(frame, landmarks):
    """The exact valid-data predicate in app.py store_l200_data, without Dash."""
    try:
        values = frame.set_index('feature')
        genes = frame['feature'].values.tolist()
        return bool(len(genes) == 200 and Counter(genes) == Counter(landmarks)
                    and len(values.columns) > 0 and not values.isnull().values.any())
    except Exception:
        return False


def selection_checks(frame, genes, model_dir):
    experiments = [c for c in frame.columns if c != 'feature']
    prepared = bench.PreparedInput(frame, experiments)
    results = []
    for gene in dict.fromkeys(genes):
        features = bench.feature_list(model_dir, gene)
        actual, positions = prepared.select(features)
        mask = frame.feature.isin(features)
        expected_positions = np.flatnonzero(mask.to_numpy())
        expected = frame[mask][experiments].to_numpy().T
        equal = (np.array_equal(positions, expected_positions) and actual.shape == expected.shape
                 and actual.dtype == expected.dtype and np.array_equal(actual, expected, equal_nan=True))
        result = {'gene': gene, 'exact': bool(equal), 'selected_input_rows': positions.tolist(),
                  'selected_input_ids': frame.iloc[positions]['feature'].tolist(),
                  'top_feature_file_entries': features, 'dtype': str(actual.dtype),
                  'shape': list(actual.shape), 'special_values': numeric_masks(actual),
                  'matrix_sha256': hashlib.sha256(actual.tobytes(order='C')).hexdigest()}
        results.append(result)
        if not equal:
            raise AssertionError('Input selection mismatch before inference: ' + gene)
    return results


def state_digest(model):
    return hashlib.sha256(pickle.dumps(model, protocol=4)).hexdigest()


def invoke(function):
    try:
        return {'accepted': True}, function()
    except Exception as exc:
        return {'accepted': False, 'exception_type': type(exc).__module__ + '.' + type(exc).__name__,
                'message': str(exc)}, None


def load_targets(path):
    raw = Path(path).read_text()
    return json.loads(raw) if raw.lstrip().startswith('[') else raw.splitlines()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--targets', type=Path)
    parser.add_argument('--hash-targets', type=Path, action='append', default=[])
    parser.add_argument('--hash-all-supported', action='store_true')
    parser.add_argument('--cache-mib', type=int, default=128)
    parser.add_argument('--deadline-seconds', type=float, default=600)
    args = parser.parse_args()
    args.repo = args.repo.resolve()
    args.results.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.deadline_seconds
    model_dir = bench.resolve_model_dir(args.models)
    canonical = list(pd.read_pickle(args.repo / 'data/19q4_genes.pkl'))
    genes = load_targets(args.targets) if args.targets else canonical[:16]
    stats, categories = bench.metadata(args.repo)
    supported, excluded = [], []
    for gene in canonical:
        reasons = []
        if not (model_dir / ('model_rd10_' + gene + '.pkl')).is_file(): reasons.append('missing_model')
        if not (model_dir / ('feats_' + gene + '.csv')).is_file(): reasons.append('missing_feature_file')
        if gene not in stats.index: reasons.append('missing_statistics')
        if reasons: excluded.append({'gene': gene, 'reasons': reasons})
        else: supported.append(gene)
    if any(g not in supported for g in genes):
        raise ValueError('Validation targets include unsupported genes.')
    policy = {'atol': ATOL, 'rtol': RTOL, 'seed': SEED,
              'declared_before_inference': True,
              'primary_goal': 'exact match; fixed tolerance retained as separate evidence',
              'canonical_count': len(canonical), 'supported_count': len(supported), 'exclusions': excluded,
              'validation_targets': genes, 'source_files_are_read_only': True,
              'source_identity': subprocess.check_output(['git', '-C', str(args.repo), 'rev-parse', 'HEAD'], text=True).strip()}
    bench.atomic_json(args.results / 'validation_policy.json', policy)
    input_path = args.repo / 'data/PC9SKMELCOLO_L200_CERES_Features.csv'
    frame = pd.read_csv(input_path)
    experiments = [c for c in frame.columns if c != 'feature']
    landmarks = list(pd.read_pickle(args.repo / 'data/l200_genes.pkl'))
    numeric = frame[experiments].to_numpy()
    bench.atomic_json(args.results / 'input_manifest.json', {
        'file': str(input_path), 'sha256': bench.sha256(input_path), 'rows': len(frame),
        'columns': frame.columns.tolist(), 'experiments': experiments, 'feature_ids_in_order': frame.feature.tolist(),
        'dtypes': {c: str(d) for c, d in frame.dtypes.items()},
        'special_values': numeric_masks(numeric), 'unique_feature_ids': int(frame.feature.nunique(dropna=False)),
        'matches_landmark_pickle_order': frame.feature.tolist() == landmarks,
        'passes_original_upload_gate': upload_gate(frame, landmarks)})
    metadata_names = ['README.md', 'requirements.txt', 'Dockerfile', 'LICENSE', 'app.py', 'app_session.py',
                      'data/19q4_sum_stats.csv', 'data/19q4_gene_cats.csv', 'data/19q4_genes.pkl',
                      'data/l200_genes.pkl', 'data/L200_landmark_genes.csv',
                      'data/PC9SKMELCOLO_L200_CERES_Features.csv',
                      'data/PC9SKMELCOLO_L200_CERES_Features_prediction.csv']
    metadata_before = [
        {'file': name, 'bytes': (args.repo / name).stat().st_size, 'sha256': bench.sha256(args.repo / name)}
        for name in metadata_names]
    bench.atomic_json(args.results / 'metadata_manifest.json', metadata_before)
    _, author = bench.load_author(args.repo, model_dir, genes, args.results)
    timer = bench.Timer()
    cache = bench.ModelCache(args.repo, model_dir, genes, timer, deadline=deadline, max_bytes=args.cache_mib * 1024**2)
    failures = []
    results = []
    original_tables = {}
    initial_sources = {g: bench.sha256(model_dir / ('model_rd10_' + g + '.pkl')) for g in set(genes)}

    def run_case(name, data, scope, original_rejected_expected=False):
        bench.check_deadline(deadline)
        author.__globals__['_goloco_scope'] = list(scope)
        known = all(g in supported for g in scope)
        selections = selection_checks(data, scope, model_dir) if known else []
        before = {}
        if known:
            for gene in dict.fromkeys(scope):
                if gene not in initial_sources:
                    initial_sources[gene] = bench.sha256(model_dir / ('model_rd10_' + gene + '.pkl'))
                before[gene] = state_digest(cache.get_model(gene))
        cache_before = cache.snapshot()
        original_status, reference = invoke(lambda: bench.original_output(author, data, deadline))
        candidate_status, candidate = invoke(lambda: bench.instrumented(data, scope, args.repo, model_dir, True, cache, deadline)[0])
        entry = {'case': name, 'targets': list(scope), 'original_upload_gate': upload_gate(data, landmarks),
                 'selection_matrices': selections, 'original': original_status, 'candidate': candidate_status,
                 'cache_before': cache_before, 'cache_after': cache.snapshot()}
        accepted = (original_status['accepted'] and candidate_status['accepted'])
        if reference is not None and candidate is not None:
            entry['comparison'] = frame_comparison(reference, candidate)
            accepted = accepted and entry['comparison']['accepted']
            if name in ('real_input', 'original_repeatability'):
                output_values = reference.select_dtypes(include=[np.number]).to_numpy()
                entry['real_output_all_finite'] = bool(np.isfinite(output_values).all())
                accepted = accepted and entry['real_output_all_finite']
            original_csv, candidate_csv = reference.to_csv(index=True), candidate.to_csv(index=True)
            entry['serialized_csv_exact'] = original_csv == candidate_csv
            original_roundtrip = pd.read_csv(io.StringIO(original_csv))
            candidate_roundtrip = pd.read_csv(io.StringIO(candidate_csv))
            entry['serialized_columns_with_index'] = original_roundtrip.columns.tolist()
            entry['serialized_output_schema_exact'] = (original_roundtrip.columns.tolist() == candidate_roundtrip.columns.tolist()
                                                       and original_roundtrip.shape == candidate_roundtrip.shape
                                                       and original_roundtrip.dtypes.equals(candidate_roundtrip.dtypes))
            accepted = accepted and entry['serialized_output_schema_exact']
            original_tables[name] = reference
            if name == 'real_input':
                reference.to_csv(args.results / 'fresh_original_16.csv', index=True)
                candidate.to_csv(args.results / 'fresh_optimized_16.csv', index=True)
        elif reference is None and candidate is None:
            entry['explicit_rejection_both'] = True
            entry['exception_types_match'] = original_status['exception_type'] == candidate_status['exception_type']
            accepted = bool(original_rejected_expected)
        else:
            entry['explicit_rejection_both'] = False
        if original_rejected_expected:
            accepted = accepted and not original_status['accepted'] and not candidate_status['accepted']
        unchanged = {}
        for gene, digest in before.items():
            if gene in cache.models:
                unchanged[gene] = digest == state_digest(cache.models[gene])
        entry['retained_model_states_unchanged'] = unchanged
        accepted = accepted and all(unchanged.values())
        entry['passed'] = bool(accepted)
        results.append(entry)
        bench.append_jsonl(args.results / 'edge_cases.jsonl', entry)
        if not accepted: failures.append(name)
        print(json.dumps({'validation_case': name, 'passed': bool(accepted)}), flush=True)

    try:
        run_case('real_input', frame, genes)
        run_case('original_repeatability', frame, genes)
        repeatability = frame_comparison(original_tables['real_input'], original_tables['original_repeatability'])
        bench.atomic_json(args.results / 'original_repeatability.json', repeatability)
        if not repeatability['accepted']: failures.append('original_repeatability_pair')
        run_case('reversed_experiment_columns', frame[['feature'] + experiments[::-1]], genes)
        rng = np.random.RandomState(SEED)
        run_case('seeded_shuffled_feature_rows', frame.iloc[rng.permutation(len(frame))].reset_index(drop=True), genes)
        changed = frame.copy()
        changed[experiments] = changed[experiments] + 0.01
        run_case('changed_numeric_values_same_ids', changed, genes)
        selected = selection_checks(frame, [genes[0]], model_dir)[0]['selected_input_rows'][0]
        duplicate = pd.concat([frame, frame.iloc[[selected]]], ignore_index=True)
        run_case('duplicate_required_feature_row', duplicate, [genes[0]], True)
        missing = frame.drop(frame.index[selected]).reset_index(drop=True)
        run_case('missing_required_feature_row', missing, [genes[0]], True)
        nan_frame = frame.copy(); nan_frame.loc[selected, experiments[0]] = np.nan
        run_case('nan_required_input', nan_frame, [genes[0]], True)
        inf_frame = frame.copy(); inf_frame.loc[selected, experiments[0]] = np.inf
        run_case('infinite_required_input', inf_frame, [genes[0]], True)
        changed_targets = list(dict.fromkeys(genes[-4:] + [g for g in supported if g not in genes][:12]))
        run_case('changed_target_set', frame, changed_targets)
        run_case('repeated_target_ids', frame, [genes[0], genes[0], genes[1]])
        run_case('unsupported_target_explicit_error', frame, ['__GOLOCO_UNSUPPORTED_TARGET__'], True)
        state_before_invalidation = state_digest(cache.get_model(genes[0]))
        snapshot_before = cache.snapshot()
        cache.invalidate(genes[0])
        state_after_invalidation = state_digest(cache.get_model(genes[0]))
        invalidation = {'target': genes[0], 'before': snapshot_before, 'after': cache.snapshot(),
                        'reloaded_model_state_equal': state_before_invalidation == state_after_invalidation,
                        'source_model_sha256': bench.sha256(model_dir / ('model_rd10_' + genes[0] + '.pkl')),
                        'policy': 'explicit invalidate(gene) reloads unchanged author artifact; no source mutation'}
        bench.atomic_json(args.results / 'cache_invalidation.json', invalidation)
        if not invalidation['reloaded_model_state_equal']: failures.append('cache_invalidation')
        run_case('request_after_explicit_invalidation', frame, genes)
        cache.invalidate()
        run_case('request_after_full_cache_invalidation', frame, genes)
        regular_cache = cache
        small_budget = max(max((model_dir / ('model_rd10_' + g + '.pkl')).stat().st_size * 2, 65536)
                           for g in genes[:2])
        cache = bench.ModelCache(args.repo, model_dir, genes[:2], bench.Timer(), deadline=deadline, max_bytes=small_budget)
        run_case('bounded_cache_eviction_recomputes', frame, [genes[0], genes[1], genes[0]])
        eviction_snapshot = cache.snapshot()
        eviction_passed = (eviction_snapshot['evictions'] >= 2
                           and eviction_snapshot['retained_accounted_bytes'] <= small_budget)
        bench.atomic_json(args.results / 'cache_eviction.json', {'passed': eviction_passed, 'snapshot': eviction_snapshot})
        if not eviction_passed: failures.append('bounded_cache_eviction')
        cache = regular_cache
        fixture_path = args.repo / 'data/PC9SKMELCOLO_L200_CERES_Features_prediction.csv'
        fixture = pd.read_csv(fixture_path, index_col=0).set_index('gene', drop=False).loc[genes].reset_index(drop=True)
        # A full-genome CSV's saved row numbers differ from a selected subset.
        # Align that presentation index explicitly; all numerical values stay intact.
        fixture.index = original_tables['real_input'].index.copy()
        fixture_comparison = frame_comparison(fixture, original_tables['real_input'])
        fixture_comparison['role'] = 'additional released-fixture sanity check; never an inference input'
        fixture_comparison['alignment'] = 'select requested gene order; reset full-genome saved row numbers to subset reference index'
        fixture_comparison['fixture_sha256'] = bench.sha256(fixture_path)
        bench.atomic_json(args.results / 'released_fixture_comparison.json', fixture_comparison)
        if not fixture_comparison['accepted']: failures.append('released_fixture_comparison')
        used = set(genes) | set(changed_targets)
        for path in args.hash_targets: used.update(load_targets(path))
        if args.hash_all_supported: used.update(supported)
        manifest_genes = [g for g in canonical if g in used]
        manifest_path = args.results / 'used_model_feature_manifest.jsonl'
        with manifest_path.open('w') as manifest_handle:
            for position, gene in enumerate(manifest_genes):
                bench.check_deadline(deadline)
                model_path = model_dir / ('model_rd10_' + gene + '.pkl')
                feature_path = model_dir / ('feats_' + gene + '.csv')
                record = {'gene': gene,
                    'model_file': str(model_path), 'model_bytes': model_path.stat().st_size, 'model_sha256': bench.sha256(model_path),
                    'feature_file': str(feature_path), 'feature_bytes': feature_path.stat().st_size, 'feature_sha256': bench.sha256(feature_path)}
                manifest_handle.write(json.dumps(record) + '\n')
                if position % 128 == 0:
                    manifest_handle.flush()
        unchanged_files = {gene: digest == bench.sha256(model_dir / ('model_rd10_' + gene + '.pkl')) for gene, digest in initial_sources.items()}
        unchanged_metadata = {item['file']: item['sha256'] == bench.sha256(args.repo / item['file']) for item in metadata_before}
        if not all(unchanged_files.values()): failures.append('source_model_files_changed')
        if not all(unchanged_metadata.values()): failures.append('source_metadata_files_changed')
        bench.atomic_json(args.results / 'source_integrity.json', {'checked_targets': list(initial_sources), 'source_model_files_unchanged': unchanged_files,
                          'source_metadata_files_unchanged': unchanged_metadata,
                          'metadata_note': 'all source paths read only; metadata hashes in metadata_manifest.json'})
        bench.atomic_json(args.results / 'summary.json', {'state': 'complete', 'passed': not failures, 'failures': failures,
                          'case_count': len(results), 'manifest_targets': len(manifest_genes), 'cache_final': cache.snapshot(),
                          'atol': ATOL, 'rtol': RTOL, 'seed': SEED})
        if failures: raise AssertionError('Supplemental validation failed: ' + ', '.join(failures))
    except Exception as exc:
        bench.atomic_json(args.results / 'status.json', {'state': 'failed', 'error': repr(exc), 'traceback': traceback.format_exc(),
                          'completed_cases': len(results), 'failures': failures})
        raise
    else:
        bench.atomic_json(args.results / 'status.json', {'state': 'complete', 'completed_cases': len(results), 'passed': True})


if __name__ == '__main__':
    main()
