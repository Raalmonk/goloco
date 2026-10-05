#!/usr/bin/env python3
"""Remote-only independent RF/B3/Rust semantics and boundary validation."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time
import traceback

import numpy as np
import pandas as pd

from goloco_benchmark import atomic_json, append_jsonl, check_deadline, feature_list, sha256
from native_candidate import FastForest
from rust_candidate import RustLibrary, RustForest

ATOL, RTOL = 1e-8, 1e-7


def digest(model):
    return hashlib.sha256(pickle.dumps(model, protocol=4)).hexdigest()


def compare(reference, candidate):
    same_shape = reference.shape == candidate.shape
    same_dtype = reference.dtype == candidate.dtype
    if not same_shape:
        return {'passed': False, 'shape_exact': False}
    finite = np.isfinite(reference) & np.isfinite(candidate)
    delta = np.abs(reference[finite] - candidate[finite])
    denominator = np.abs(reference[finite])
    rel = np.divide(delta, denominator, out=np.zeros_like(delta), where=denominator != 0)
    rel[(denominator == 0) & (delta != 0)] = np.inf
    masks = {}
    for name, function in [('nan', np.isnan), ('positive_inf', np.isposinf), ('negative_inf', np.isneginf)]:
        masks[name] = {'reference': np.argwhere(function(reference)).tolist(),
                       'candidate': np.argwhere(function(candidate)).tolist()}
    same_masks = all(x['reference'] == x['candidate'] for x in masks.values())
    exact = bool(np.array_equal(reference, candidate, equal_nan=True))
    close = bool(np.allclose(reference, candidate, atol=ATOL, rtol=RTOL, equal_nan=True))
    return {'passed': exact and same_dtype and same_masks,
            'shape_exact': same_shape, 'dtype_exact': same_dtype,
            'shape': list(candidate.shape), 'dtype': str(candidate.dtype),
            'exact_match': exact, 'within_predeclared_tolerances': close,
            'atol': ATOL, 'rtol': RTOL, 'special_value_masks': masks,
            'max_absolute_error': float(delta.max()) if delta.size else 0.0,
            'max_relative_error': float(rel.max()) if rel.size else 0.0}


def outcome(function, matrix):
    try:
        return {'accepted': True}, function(matrix)
    except Exception as exc:
        return {'accepted': False, 'type': type(exc).__module__ + '.' + type(exc).__name__,
                'message': str(exc)}, None


def root_threshold_probes(model, base):
    """Roots are always reached; float32 brackets must reach distinct leaves."""
    rows, details = [], []
    for tree_number, estimator in enumerate(model.estimators_):
        tree = estimator.tree_
        if tree.children_left[0] == -1:
            continue
        feature = int(tree.feature[0])
        threshold = np.float64(tree.threshold[0])
        rounded = np.float32(threshold)
        lower = rounded if np.float64(rounded) <= threshold else np.nextafter(rounded, np.float32(-np.inf))
        upper = np.nextafter(lower, np.float32(np.inf))
        assert np.float64(lower) <= threshold < np.float64(upper)
        values = [np.float64(lower), np.float64(upper), threshold,
                  np.nextafter(threshold, np.float64(-np.inf)),
                  np.nextafter(threshold, np.float64(np.inf))]
        start = len(rows)
        for value in values:
            row = np.array(base, dtype=np.float64, copy=True)
            row[feature] = value
            rows.append(row)
        bracket = np.asarray(rows[start:start + 2], dtype=np.float32)
        leaves = tree.apply(bracket)
        assert leaves[0] != leaves[1], 'Root threshold brackets did not take different branches'
        details.append({'tree': tree_number, 'root_feature': feature,
                        'threshold_float64_hex': float(threshold).hex(),
                        'input_row_offset': start, 'probe_count': len(values),
                        'lower_float32_as_float64': float(lower),
                        'upper_float32_as_float64': float(upper),
                        'lower_cast_goes_left': True, 'upper_cast_goes_right': True,
                        'distinct_leaf_ids': leaves.tolist(),
                        'raw_probe_float64_hex': [float(v).hex() for v in values],
                        'actual_probe_float32_as_float64': [float(np.float32(v)) for v in values]})
    return np.asarray(rows, dtype=np.float64), details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--tier-results', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--models', type=int, default=16)
    parser.add_argument('--deadline-seconds', type=float, default=240)
    args = parser.parse_args()
    if not Path('/content').is_dir():
        raise RuntimeError('This validation runs only inside the dedicated Colab runtime.')
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'cases.jsonl').exists():
        raise RuntimeError('Preserve completed records; choose a distinct validation output path.')
    deadline = time.monotonic() + args.deadline_seconds
    protocol = json.loads((args.tier_results / 'protocol.json').read_text())
    model_dir = Path(protocol['model_dir'])
    genes = protocol['targets'][:args.models]
    frame = pd.read_csv(protocol['input'])
    experiments = protocol['experiments']
    atomic_json(args.out / 'policy.json', {'atol': ATOL, 'rtol': RTOL, 'require_exact': True,
                'declared_before_model_loading': True, 'models': genes,
                'source_protocol_sha256': sha256(args.tier_results / 'protocol.json'),
                'library_sha256': sha256(args.library), 'input_sha256': sha256(Path(protocol['input'])),
                'root_probe_policy': 'Every internal root of each tested forest; adjacent float32 values straddling unchanged float64 threshold; distinct leaf IDs verified',
                'prediction_output_is_never_cached': True})
    library = RustLibrary(args.library)
    failures, totals, states = [], {'valid_cases': 0, 'invalid_cases': 0, 'root_thresholds': 0, 'probe_rows': 0}, []
    try:
        for gene in genes:
            check_deadline(deadline)
            source = model_dir / ('model_rd10_' + gene + '.pkl')
            source_before = sha256(source)
            with source.open('rb') as handle:
                original = pickle.load(handle)
            state_before = digest(original)
            b3, rust = FastForest(original), RustForest(original, library)
            assert b3.estimator is original and rust.estimator is original
            features = feature_list(model_dir, gene)
            matrix = frame[frame.feature.isin(features)][experiments].to_numpy().T
            matrix32 = original._validate_X_predict(matrix)
            root_probes, threshold_details = root_threshold_probes(original, matrix[0])
            atomic_json(args.out / ('thresholds_' + gene + '.json'), threshold_details)
            totals['root_thresholds'] += len(threshold_details)
            totals['probe_rows'] += len(root_probes)
            valid = [('real_matrix', matrix),
                     ('reordered_experiment_rows', matrix[::-1]),
                     ('changed_numeric_values', matrix + 0.01),
                     ('float32_fortran_layout', np.asfortranarray(matrix32)),
                     ('float32_negative_sample_stride', matrix32[::-1]),
                     ('float32_negative_feature_stride', matrix32[:, ::-1]),
                     ('float64_fortran_layout', np.asfortranarray(matrix))]
            if len(root_probes): valid.append(('float32_root_threshold_brackets', root_probes))
            for name, data in valid:
                check_deadline(deadline)
                reference = original.predict(data)
                comparisons = {'B3': compare(reference, b3.predict(data)),
                               'Rust': compare(reference, rust.predict(data))}
                passed = all(value['passed'] for value in comparisons.values())
                append_jsonl(args.out / 'cases.jsonl', {'gene': gene, 'case': name,
                             'input_shape': list(data.shape), 'input_dtype': str(data.dtype),
                             'input_byte_strides': list(data.strides), 'valid_input': True,
                             'comparisons': comparisons, 'passed': passed})
                totals['valid_cases'] += 1
                if not passed: failures.append(gene + ':' + name)
            invalid = [('wrong_feature_count', matrix[:, :-1]),
                       ('one_dimensional_input', matrix[0]),
                       ('empty_sample_matrix', matrix[:0]),
                       ('three_dimensional_input', matrix[np.newaxis, :, :])]
            for name, value in [('nan', np.nan), ('positive_infinity', np.inf), ('negative_infinity', -np.inf)]:
                changed = matrix.copy(); changed[0, 0] = value
                invalid.append((name, changed))
            for name, data in invalid:
                check_deadline(deadline)
                responses = {label: outcome(implementation.predict, data)[0]
                             for label, implementation in [('original', original), ('B3', b3), ('Rust', rust)]}
                passed = (not any(r['accepted'] for r in responses.values())
                          and len(set(r.get('type') for r in responses.values())) == 1)
                append_jsonl(args.out / 'cases.jsonl', {'gene': gene, 'case': name,
                             'valid_input': False, 'responses': responses, 'passed': passed})
                totals['invalid_cases'] += 1
                if not passed: failures.append(gene + ':' + name)
            state = {'gene': gene, 'model_state_before_sha256': state_before,
                     'model_state_after_sha256': digest(original),
                     'source_file_before_sha256': source_before, 'source_file_after_sha256': sha256(source),
                     'exported_buffer_bytes': rust.packed_bytes,
                     'exported_buffers_readonly': {name: not array.flags.writeable for name, array in rust.arrays.items()}}
            state['passed'] = (state['model_state_before_sha256'] == state['model_state_after_sha256']
                               and state['source_file_before_sha256'] == state['source_file_after_sha256']
                               and all(state['exported_buffers_readonly'].values()))
            states.append(state)
            atomic_json(args.out / 'state_integrity.json', states)
            if not state['passed']: failures.append(gene + ':state_integrity')
            print(json.dumps({'native_edge_model': gene, 'root_thresholds': len(threshold_details), 'failures': len(failures)}), flush=True)
        atomic_json(args.out / 'summary.json', {'status': 'complete', 'passed': not failures,
                    'tested_models': len(genes), 'totals': totals, 'failures': failures,
                    'every_valid_comparison_exact': not failures, 'atol': ATOL, 'rtol': RTOL})
        if failures: raise AssertionError('Native edge validation failures: ' + ', '.join(failures))
    except Exception as exc:
        atomic_json(args.out / 'status.json', {'status': 'failed', 'error': repr(exc),
                    'traceback': traceback.format_exc(), 'totals': totals, 'failures': failures})
        raise
    else:
        atomic_json(args.out / 'status.json', {'status': 'complete', 'passed': True, 'totals': totals})


if __name__ == '__main__':
    main()
