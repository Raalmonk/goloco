#!/usr/bin/env python3
"""Optional balanced, matched-residency chunk comparison on the authorized VM."""
import argparse
import ctypes
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'reference'))


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['repo', 'model-root', 'input', 'targets', 'library', 'oracle', 'out']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--bundle-mode', choices=['mmap', 'load'], default='mmap')
    parser.add_argument('--chunks', default='1,64,256')
    parser.add_argument('--reps', type=int, default=5)
    parser.add_argument('--deadline-epoch', type=float, required=True)
    parser.add_argument('--process-limit-bytes', type=int, required=True)
    parser.add_argument('--max-cache-bytes', type=int, default=16 * 1024 ** 3)
    return parser.parse_args()


def descriptor_rejections(service, frame, gene):
    """Malformed arguments fail in Rust before traversal; original arrays intact."""
    import numpy as np
    from goloco_reusable.backend import PreparedInput
    from goloco_reusable.chunk import ForestRequest
    experiments = [column for column in frame.columns if column != 'feature']
    with service._operation():
        entry = service._get_entry(gene)
        matrix, _ = PreparedInput(frame, experiments).select(entry['features'])
        checked = entry['model']._validate_X_predict(matrix)
        forest = entry['predictor']
        output = np.empty(checked.shape[0], dtype=np.float64)
        low, high = np.byte_bounds(checked)
        arrays = [forest.arrays[name] for name in ['left', 'right', 'feature', 'threshold', 'value', 'roots']]
        descriptor = ForestRequest(*[array.ctypes.data for array in arrays],
            *[array.size for array in arrays], forest.n_nodes, forest.n_trees,
            checked.ctypes.data, low, high, checked.shape[0], checked.shape[1],
            checked.strides[0], checked.strides[1], output.ctypes.data, output.size)
        failed = ctypes.c_size_t()
        descriptor.left_len -= 1
        status_length = service._chunk_library.predict(ctypes.byref(descriptor), 1, ctypes.byref(failed))
        assert status_length == 31 and failed.value == 0
        descriptor.left_len += 1
        descriptor.input_high = descriptor.input_low
        status_bounds = service._chunk_library.predict(ctypes.byref(descriptor), 1, ctypes.byref(failed))
        assert status_bounds == 33 and failed.value == 0
        descriptor.input_high = high
        maximum = ctypes.c_size_t(-1).value
        overflow_statuses = {}
        for kind in ['node_length', 'tree_length', 'output_length', 'array_pointer', 'output_pointer']:
            mutant = ForestRequest.from_buffer_copy(descriptor)
            if kind == 'node_length':
                mutant.n_nodes = maximum
                mutant.left_len = mutant.right_len = mutant.feature_len = maximum
                mutant.threshold_len = mutant.value_len = maximum
            elif kind == 'tree_length':
                mutant.n_trees = mutant.roots_len = maximum
            elif kind == 'output_length':
                mutant.output_len = maximum
            elif kind == 'array_pointer':
                mutant.left = maximum - 3
            else:
                mutant.output = maximum - 3
            status = service._chunk_library.predict(ctypes.byref(mutant), 1, ctypes.byref(failed))
            assert status == 34 and failed.value == 0, (kind, status, failed.value)
            overflow_statuses[kind] = status
        return {'length_mismatch_status': status_length, 'input_bounds_status': status_bounds,
                'overflow_statuses': overflow_statuses, 'native_rejection_before_traversal': True}


def optional_correctness(service, frame, genes):
    """Exact original-estimator probes plus order/value tests against chunk1."""
    import numpy as np
    import pandas as pd
    from goloco_reusable.backend import PreparedInput
    from goloco_reusable.chunk import predict_checked_chunk
    sample = genes[:16]
    experiments = [column for column in frame.columns if column != 'feature']
    prepared = PreparedInput(frame, experiments)
    forests, inputs = [], []
    with service._operation():
        for gene in sample:
            entry = service._get_entry(gene)
            matrix, _ = prepared.select(entry['features'])
            checked = entry['model']._validate_X_predict(matrix)
            forests.append(entry['predictor'])
            inputs.append(checked)
        matrix_cases = {
            'real': inputs,
            'negative_sample_stride': [value[::-1] for value in inputs],
            'negative_feature_stride': [value[:, ::-1] for value in inputs],
            'fortran': [np.asfortranarray(value) for value in inputs],
            'changed_values': [value * np.float32(.875) + np.float32(.125) for value in inputs],
        }
        for name, matrices in matrix_cases.items():
            result, metrics = predict_checked_chunk(service._chunk_library, forests, matrices)
            for forest, matrix, actual in zip(forests, matrices, result):
                # Primary direct sklearn estimator, with its unchanged validator.
                expected = forest.estimator.predict(matrix)
                np.testing.assert_array_equal(expected, actual, err_msg=name)
            assert metrics['ffi_calls'] == 1
    changed = frame.copy(deep=True)
    changed[experiments] = changed[experiments] * .875 + .125
    frames = {'real': frame, 'changed_values': changed,
              'experiment_order': frame[['feature'] + list(reversed(experiments))]}
    ordered = sample + [sample[0], sample[-1]]
    for name, value in frames.items():
        service.chunk_size = 1
        expected = service.predict(value, ordered)
        for chunk in [64, 256]:
            service.chunk_size = chunk
            actual = service.predict(value, ordered)
            pd.testing.assert_frame_equal(expected, actual, check_exact=True)
            assert expected.to_csv() == actual.to_csv(), name
    service.chunk_size = 1
    return {'target_models': len(sample), 'matrix_cases': list(matrix_cases),
            'matrix_reference': 'original sklearn RandomForestRegressor.predict',
            'request_cases': list(frames), 'request_reference': 'unchanged core chunk1',
            'repeated_target_order_checked': True, 'all_exact': True}


def contract_regressions(service, frame, genes, out):
    """Untimed selected/unused invalid-value and first-error contract checks."""
    import numpy as np
    import pandas as pd
    from goloco_benchmark import append_jsonl
    sample = genes[:16]
    experiments = [column for column in frame.columns if column != 'feature']
    selected_features = set(service._entries[sample[0]]['features'])
    all_features = set().union(*(set(service._entries[gene]['features']) for gene in sample))
    selected = next(i for i, value in enumerate(frame.feature) if value in selected_features)
    unused = next(i for i, value in enumerate(frame.feature) if value not in all_features)
    changed = frame.copy(deep=True)
    changed.loc[:, experiments] += .01
    cases = [('real',frame,sample),('changed_values',changed,sample),
             ('repeated_targets',frame,sample+[sample[0],sample[-1]]),
             ('reversed_experiments',frame[['feature']+experiments[::-1]],sample),
             ('reversed_feature_rows',frame.iloc[::-1].copy(),sample)]
    for label, position in [('selected',selected),('unused',unused)]:
        for value, suffix in [(np.nan,'nan'),(np.inf,'positive_inf'),(-np.inf,'negative_inf')]:
            altered = frame.copy(deep=True)
            altered.iloc[position,altered.columns.get_loc(experiments[0])] = value
            cases.append((label+'_'+suffix,altered,sample))
        cases.append(('missing_'+label+'_feature',frame.drop(frame.index[position]),sample))
        cases.append(('duplicate_'+label+'_feature',pd.concat([frame,frame.iloc[[position]]],ignore_index=True),sample))
    unknown = '__GOLOCO_UNKNOWN_TARGET__'
    invalid = frame.copy(deep=True)
    invalid.iloc[selected,invalid.columns.get_loc(experiments[0])] = np.nan
    cases += [('first_invalid_feature_then_unsupported',invalid,[sample[0],unknown]),
              ('first_unsupported_then_invalid_feature',invalid,[unknown,sample[0]]),
              ('valid_pending_then_unsupported',frame,[sample[0],unknown]),
              ('unsupported_before_valid',frame,[unknown,sample[0]])]
    def invoke(data, scope):
        diagnostics = {}
        try:
            value = service.predict(data,scope,diagnostics=diagnostics)
            return {'accepted':True,'diagnostics':diagnostics},value
        except Exception as exc:
            return {'accepted':False,'exception_type':type(exc).__module__+'.'+type(exc).__name__,
                    'message':str(exc),'diagnostics':diagnostics},None
    records = 0
    try:
        for name, data, scope in cases:
            service.chunk_size = 1
            expected_status, expected = invoke(data,scope)
            for chunk in [64,256]:
                service.chunk_size = chunk
                actual_status, actual = invoke(data,scope)
                equal = actual_status['accepted'] == expected_status['accepted']
                if equal and expected_status['accepted']:
                    pd.testing.assert_frame_equal(expected,actual,check_exact=True)
                    equal = expected.to_csv() == actual.to_csv()
                    equal = equal and actual_status['diagnostics']['estimator_calls']==len(scope)
                elif equal:
                    equal = actual_status['exception_type']==expected_status['exception_type']
                row = {'case':name,'chunk_size':chunk,'targets':scope,'reference':expected_status,
                       'candidate':actual_status,'passed':equal,'reference_backend':'unchanged core chunk1'}
                append_jsonl(out/'contract_cases.jsonl',row)
                records += 1
                assert equal, (name,chunk,expected_status,actual_status)
                snapshot = service.snapshot()
                assert snapshot['pinned_entries']==0 and snapshot['inflight_buffer_accounted_bytes']==0
    finally:
        service.chunk_size = 1
    return {'cases':len(cases),'comparisons':records,'models':len(sample),'all_exact_or_same_exception_type':True,
            'selected_row':selected,'unused_row':unused,'first_error_order_checked':True,
            'pending_owners_released_after_success_and_error':True}


def pressure_regression(service, frame, genes, args):
    """Untimed small source-only cache exercises pending-owner pinning and LRU."""
    import gc
    import pandas as pd
    from goloco_reusable.chunk import ChunkedGoloco
    # Derive source-mode entry charges even when the shared comparison uses a
    # mapped bundle; source mode additionally owns each forest's exported arrays.
    charges = {}
    for gene in genes[:16]:
        entry = service._entries[gene]
        predictor = entry['predictor']
        charges[gene] = entry['charge'] + (0 if predictor.owns_export else predictor.packed_bytes)
    ranked = sorted(charges,key=lambda gene:(charges[gene],gene))
    assert len(ranked)>=4
    chosen = [ranked[index] for index in [0,len(ranked)//3,2*len(ranked)//3,len(ranked)-1]]
    assert len(set(chosen))==4
    sizes = sorted([charges[gene] for gene in chosen],reverse=True)
    metadata = service.snapshot()['metadata_bytes']
    budget = metadata + sum(sizes[:2]) + len(chosen)*4096 + 32768
    assert sum(sizes)>sum(sizes[:2])+32768, 'Fixture must force a partial chunk'
    assert service.snapshot()['aggregate_cache_accounted_bytes']+budget<=args.max_cache_bytes
    scope = chosen+[chosen[0],chosen[-1]]
    service.chunk_size = 1
    expected = service.predict(frame,scope)
    stressed = ChunkedGoloco(args.repo,args.model_root,chunk_size=64,library=args.library,
                            model_bundle=None,max_cache_bytes=budget,
                            process_limit_bytes=args.process_limit_bytes)
    try:
        before = stressed.snapshot()
        diagnostics = {}
        actual = stressed.predict(frame,scope,diagnostics=diagnostics)
        pd.testing.assert_frame_equal(expected,actual,check_exact=True)
        assert expected.to_csv()==actual.to_csv()
        after = stressed.snapshot()
        assert after['chunk_budget_flushes']>before['chunk_budget_flushes']
        assert after['evictions']>before['evictions']
        assert diagnostics['estimator_calls']==len(scope)
        assert sum(diagnostics['actual_chunk_sizes'])==len(scope)
        assert max(diagnostics['actual_chunk_sizes'])<64
        assert after['aggregate_cache_accounted_bytes']<=budget
        assert after['pinned_entries']==0 and after['inflight_buffer_accounted_bytes']==0
        return {'passed':True,'reference_backend':'unchanged core chunk1','exact_frame_and_csv':True,
            'ordinary_source_without_bundle':True,'chosen_targets':chosen,'request_targets':scope,
            'entry_charge_bytes':{gene:charges[gene] for gene in chosen},'budget_bytes':budget,
            'budget_policy':'metadata + two largest selected source/adapter entries + identity ledger + 32KiB workspace',
            'before':before,'after':after,'diagnostics':diagnostics,
            'pinned_owner_pressure_proof':'At least one budget flush occurred before pinned entries became eligible for eviction',
            'pending_owners_released':True}
    finally:
        stressed.close()
        del stressed
        gc.collect()


def main():
    if not Path('/content/goloco_perf').is_dir():
        raise RuntimeError('Run optional scientific comparisons only in the dedicated Colab runtime')
    args = arguments()
    args.out.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import pandas as pd
    from goloco_benchmark import Memory, atomic_json, append_jsonl, sha256
    from goloco_experiment import correctness
    from goloco_reusable.chunk import ChunkedGoloco
    from run_phase2 import faults, subtract, summary, targets
    genes = targets(args.targets)
    chunks = [int(value) for value in args.chunks.split(',')]
    if not chunks or 1 not in chunks or len(chunks) != len(set(chunks)) or any(not 1 <= n <= 4096 for n in chunks):
        raise ValueError('Provide distinct valid chunk sizes including the unchanged chunk1 control')
    protocol = json.loads((args.oracle / 'protocol.json').read_text())
    if protocol['targets'] != genes or protocol['input_sha256'] != sha256(args.input):
        raise ValueError('Oracle target order/input identity differs from this experiment')
    expected = pd.read_pickle(args.oracle / 'oracle_real.pkl')
    oracle_hash = sha256(args.oracle / 'oracle_real.csv')
    experiments = [column for column in pd.read_csv(args.input, nrows=0).columns if column != 'feature']
    atomic_json(args.out / 'protocol.json', {
        'targets': genes, 'input_sha256': sha256(args.input), 'library_sha256': sha256(args.library),
        'oracle': str(args.oracle), 'chunks': chunks, 'reps': args.reps, 'seed': 20261005,
        'boundary': 'CSV parse through closed CSV output, no fsync',
        'memory_class': 'one shared fully prepared backend; identical source models and immutable arrays for every arm',
        'bundle': str(args.bundle) if args.bundle else None, 'bundle_mode': args.bundle_mode if args.bundle else None,
        'deadline_epoch': args.deadline_epoch, 'aggregate_process_limit_bytes': args.process_limit_bytes,
        'cache_limit_bytes': args.max_cache_bytes,
        'allocation_counter_scope': 'chunk1 core counts validation copies/output; chunked adds actual selection and descriptor allocations',
    })
    if time.time() >= args.deadline_epoch - 30:
        atomic_json(args.out / 'status.json', {'state': 'skipped', 'reason': 'insufficient session budget'})
        return
    records, orders = [], []
    sink = args.out / 'same_output_sink.csv'
    with Memory() as memory:
        started, cpu = time.perf_counter(), time.process_time()
        service = ChunkedGoloco(args.repo, args.model_root, chunk_size=1, library=args.library,
            model_bundle=args.bundle, bundle_mode=args.bundle_mode, max_cache_bytes=args.max_cache_bytes,
            process_limit_bytes=args.process_limit_bytes)
        cache = service.prepare(genes)
        preload = {'wall_seconds': time.perf_counter()-started, 'cpu_seconds': time.process_time()-cpu,
                   'cache': cache}
    preload['memory'] = memory.result()
    atomic_json(args.out / 'preload.json', preload)
    try:
        assert cache['retained_models'] == len(genes)
        assert cache['aggregate_cache_accounted_bytes'] <= args.max_cache_bytes
        assert memory.peak <= args.process_limit_bytes
        atomic_json(args.out / 'descriptor_rejections.json', descriptor_rejections(service, pd.read_csv(args.input), genes[0]))
        atomic_json(args.out / 'optional_correctness.json', optional_correctness(service, pd.read_csv(args.input), genes))
        atomic_json(args.out / 'contract_summary.json', contract_regressions(service,pd.read_csv(args.input),genes,args.out))
        atomic_json(args.out / 'cache_pressure_regression.json', pressure_regression(service,pd.read_csv(args.input),genes,args))
        def run(chunk, repetition, measured):
            if time.time() >= args.deadline_epoch - 20:
                return False
            service.chunk_size = chunk
            before, before_faults = service.snapshot(), faults()
            diagnostics = {}
            with Memory() as observed:
                started, cpu = time.perf_counter(), time.process_time()
                # Both arms include exactly the same application boundary.
                result = service.predict_csv(args.input, sink, genes, diagnostics=diagnostics)
                wall, cpu_seconds = time.perf_counter()-started, time.process_time()-cpu
            after, after_faults = service.snapshot(), faults()
            checks = correctness(expected, result, genes, experiments)
            checks['serialized_csv_bytes_exact'] = sha256(sink) == oracle_hash
            assert checks['serialized_csv_bytes_exact']
            assert diagnostics['estimator_calls'] == len(genes)
            assert after['misses'] == before['misses'] and after['evictions'] == before['evictions']
            assert after['native_exports'] == before['native_exports']
            assert after['aggregate_cache_accounted_bytes'] <= args.max_cache_bytes
            assert observed.peak <= args.process_limit_bytes
            expected_ffi = (len(genes) + chunk - 1) // chunk
            assert diagnostics['ffi_calls'] == expected_ffi
            if chunk != 1:
                assert sum(diagnostics['actual_chunk_sizes']) == len(genes)
                assert max(diagnostics['actual_chunk_sizes']) <= chunk
            row = {'arm': 'chunk' + str(chunk), 'chunk_size': chunk, 'repetition': repetition,
                   'measured': measured, 'wall_seconds': wall, 'cpu_seconds': cpu_seconds,
                   'diagnostics': diagnostics, 'correctness': checks, 'memory': observed.result(),
                   'page_faults': subtract(after_faults, before_faults), 'cache_before': before, 'cache_after': after,
                   'csv_sha256': sha256(sink), 'expected_ffi_calls': expected_ffi}
            append_jsonl(args.out / ('timings.jsonl' if measured else 'prevalidation.jsonl'), row)
            if measured:
                records.append(row)
                atomic_json(args.out / 'summary.json', summary(records))
            print(json.dumps({'event': 'chunk_complete', 'chunk': chunk, 'rep': repetition,
                              'measured': measured, 'seconds': wall, 'ffi_calls': diagnostics['ffi_calls']}), flush=True)
            return True
        complete = all(run(chunk, -1, False) for chunk in chunks)
        base = [int(value) for value in np.random.RandomState(20261005).permutation(chunks)]
        if complete:
            for repetition in range(args.reps):
                # Rotate a seeded order; for five trials every arm occupies each
                # position once or twice. All share exactly the same residency.
                offset = repetition % len(base)
                order = base[offset:] + base[:offset]
                orders.append(order)
                atomic_json(args.out / 'arm_order.json', orders)
                if not all(run(chunk, repetition, True) for chunk in order):
                    complete = False
                    break
        atomic_json(args.out / 'status.json', {'state': 'complete' if complete else 'partial_deadline',
            'samples': len(records), 'expected_samples': args.reps * len(chunks), 'epoch': time.time()})
    finally:
        service.close()


if __name__ == '__main__':
    main()
