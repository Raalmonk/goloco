#!/usr/bin/env python3
"""Independent original-oracle contract and reusable-service regressions (remote only)."""
import argparse
import gc
import hashlib
import json
import logging
from pathlib import Path
import pickle
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'reference'))


def outcome(fn):
    try:
        return {'accepted': True}, fn()
    except Exception as exc:
        return {'accepted': False, 'type': type(exc).__module__+'.'+type(exc).__name__, 'message': str(exc)}, None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--model-root', type=Path, required=True)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--targets', type=Path, required=True)
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--bundle', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--deadline-epoch', type=float, required=True)
    p.add_argument('--process-limit-bytes', type=int, required=True)
    p.add_argument('--max-cache-bytes', type=int, default=16*1024**3)
    p.add_argument('--eviction-budget-bytes', type=int)
    p.add_argument('--count', type=int, default=16)
    args = p.parse_args()
    assert Path('/content').is_dir()
    args.out.mkdir(parents=True, exist_ok=True)
    assert not (args.out/'cases.jsonl').exists()
    logging.disable(logging.CRITICAL)
    import numpy as np
    import pandas as pd
    import goloco_benchmark as ref
    from supplemental_validation import frame_comparison
    from goloco_reusable import HeadlessGoloco
    deadline = time.monotonic()+args.deadline_epoch-time.time()
    genes = json.loads(args.targets.read_text())[:args.count]
    frame = pd.read_csv(args.input)
    experiments = [c for c in frame.columns if c != 'feature']
    model_dir = ref.resolve_model_dir(args.model_root)
    paths = [model_dir/('model_rd10_'+g+'.pkl') for g in genes] + [model_dir/('feats_'+g+'.csv') for g in genes]
    paths += [args.repo/'app.py', args.repo/'app_session.py', args.repo/'data/19q4_sum_stats.csv', args.repo/'data/19q4_gene_cats.csv']
    before_hashes = {str(path): ref.sha256(path) for path in paths}
    ref.atomic_json(args.out/'policy.json', {'atol': 1e-8, 'rtol': 1e-7, 'exact_primary': True,
        'target_count': len(genes), 'targets': genes, 'input_sha256': ref.sha256(args.input),
        'semantics': 'headless author inference accepted/rejected states, not dashboard upload gate',
        'exception_equivalence': 'exact exception class; messages recorded separately because filesystem paths differ',
        'deadline_epoch': args.deadline_epoch})
    _, original = ref.load_author(args.repo, model_dir, genes, args.out, neutral_progress=True)
    first_features = set(ref.feature_list(model_dir, genes[0]))
    all_features = set().union(*(set(ref.feature_list(model_dir,g)) for g in genes))
    selected = next(i for i,v in enumerate(frame.feature) if v in first_features)
    unused = next(i for i,v in enumerate(frame.feature) if v not in all_features)
    cases = [('real', frame.copy(), genes), ('reverse_targets',frame.copy(),genes[::-1]),
             ('repeated_targets',frame.copy(),[genes[0],genes[1],genes[0]]),
             ('changed_target_subset',frame.copy(),genes[3:9]),
             ('reversed_rows',frame.iloc[::-1].copy(),genes),
             ('reversed_experiments',frame[['feature']+experiments[::-1]].copy(),genes)]
    changed = frame.copy()
    changed.loc[:, experiments] += .01
    cases.append(('changed_values',changed,genes))
    cases.append(('single_experiment',frame[['feature',experiments[0]]].copy(),genes))
    for label,position in [('selected',selected),('unused',unused)]:
        for value,name in [(np.nan,'nan'),(np.inf,'positive_inf'),(-np.inf,'negative_inf')]:
            altered = frame.copy()
            altered.iloc[position, altered.columns.get_loc(experiments[0])] = value
            cases.append((label+'_'+name,altered,genes))
        cases.append(('missing_'+label+'_row',frame.drop(frame.index[position]).copy(),genes))
        cases.append(('duplicate_'+label+'_row',pd.concat([frame,frame.iloc[[position]]],ignore_index=True),genes))
    cases.append(('missing_feature_column',frame.drop(columns=['feature']),genes))
    cases.append(('missing_experiment_columns',frame[['feature']].copy(),genes))
    duplicate_columns = frame.copy()
    duplicate_columns.columns = ['feature', experiments[0], experiments[0], experiments[2]]
    cases.append(('duplicate_experiment_columns',duplicate_columns,genes))
    cases.append(('unknown_target',frame.copy(),['__GOLOCO_UNKNOWN_TARGET__']))
    bad = frame.copy()
    bad.iloc[selected,bad.columns.get_loc(experiments[0])] = np.nan
    cases.extend([('first_error_feature_then_target',bad,[genes[0],'__GOLOCO_UNKNOWN_TARGET__']),
                  ('first_error_target_then_feature',bad,['__GOLOCO_UNKNOWN_TARGET__',genes[0]])])
    references = {}
    for name,data,scope in cases:
        ref.check_deadline(deadline)
        original.__globals__['_goloco_scope'] = list(scope)
        status, output = outcome(lambda: ref.original_output(original,data,deadline))
        references[name] = (status,output)
        ref.append_jsonl(args.out/'original_cases.jsonl',dict(case=name,targets=scope,outcome=status))
    variants = [('numpy',None,'mmap'),('python_fast',None,'mmap'),('rust',None,'mmap')]
    if args.bundle:
        variants += [('rust',args.bundle,'load'),('rust',args.bundle,'mmap')]
    failures, rows, eviction_budgets = [], [], {}
    for backend,bundle,mode in variants:
        ref.check_deadline(deadline)
        label = backend if bundle is None else 'packed_'+mode
        engine = HeadlessGoloco(args.repo,args.model_root,backend=backend,model_bundle=bundle,bundle_mode=mode,
                    library=args.library,max_cache_bytes=args.max_cache_bytes,process_limit_bytes=args.process_limit_bytes)
        try:
            engine.prepare(genes)
            model_state = {gene: hashlib.sha256(pickle.dumps(entry['model'],protocol=4)).hexdigest()
                           for gene,entry in engine._entries.items()}
            if bundle is None:
                eviction_budgets[backend] = (engine.snapshot()['metadata_bytes']
                    + max(entry['charge'] for entry in engine._entries.values()) + len(genes)*4096 + 8192)
            for name,data,scope in cases:
                ref.check_deadline(deadline)
                expected_status,expected = references[name]
                diagnostics = {}
                status,actual = outcome(lambda: engine.predict(data,scope,diagnostics=diagnostics))
                accepted_equal = expected_status['accepted'] == status['accepted']
                row = {'variant':label,'case':name,'expected':expected_status,'candidate':status,
                       'accepted_state_equal':accepted_equal,'diagnostics':diagnostics}
                if expected_status['accepted'] and status['accepted']:
                    row['comparison'] = frame_comparison(expected,actual)
                    row['passed'] = row['comparison']['exact_match'] and row['comparison']['accepted']
                    row['passed'] = row['passed'] and diagnostics['estimator_calls'] == len(scope)
                elif not expected_status['accepted'] and not status['accepted']:
                    row['exception_type_equal'] = status['type'] == expected_status['type']
                    row['passed'] = row['exception_type_equal']
                else:
                    row['passed'] = False
                ref.append_jsonl(args.out/'cases.jsonl',row)
                rows.append(row)
                if not row['passed']:
                    failures.append({'variant':label,'case':name})
            unchanged = all(hashlib.sha256(pickle.dumps(engine._entries[gene]['model'],protocol=4)).hexdigest()==checksum
                            for gene,checksum in model_state.items())
            ref.append_jsonl(args.out/'lifecycle.jsonl',{'variant':label,'operation':'estimator_state_unchanged',
                'passed':unchanged,'models':len(model_state)})
            if not unchanged:
                failures.append({'variant':label,'case':'estimator_state_unchanged'})
            engine._lock.acquire()
            try:
                overlapping,_ = outcome(lambda: engine.predict(frame,genes))
            finally:
                engine._lock.release()
            overlap_passed = not overlapping['accepted'] and overlapping['type']=='builtins.RuntimeError'
            ref.append_jsonl(args.out/'lifecycle.jsonl',{'variant':label,'operation':'single_caller_rejection',
                'passed':overlap_passed,'outcome':overlapping})
            if not overlap_passed:
                failures.append({'variant':label,'case':'single_caller_rejection'})
            original.__globals__['_goloco_scope'] = genes
            expected = references['real'][1]
            for operation in ['invalidate_one','invalidate_all','gc_lifetime','close_reopen']:
                ref.check_deadline(deadline)
                before = engine.snapshot()
                if operation == 'invalidate_one':
                    engine.invalidate(genes[0])
                elif operation == 'invalidate_all':
                    engine.invalidate()
                elif operation == 'gc_lifetime':
                    gc.collect()
                else:
                    # Exercise a retained native adapter/issued mmap view while
                    # its loader's owning service closes, then drop that view.
                    held = engine._entries[genes[0]]['predictor']
                    features = engine._entries[genes[0]]['features']
                    matrix = frame[frame.feature.isin(features)][experiments].to_numpy().T
                    before_close_values = held.predict(matrix)
                    engine.close()
                    np.testing.assert_array_equal(before_close_values,held.predict(matrix))
                    del held
                    gc.collect()
                    rejected,_ = outcome(lambda: engine.predict(frame,genes))
                    if rejected['accepted']:
                        failures.append({'variant':label,'case':'closed_service_accepted'})
                    engine.reopen()
                diagnostics = {}
                result = engine.predict(frame,genes,diagnostics=diagnostics)
                comparison = frame_comparison(expected,result)
                passed = comparison['exact_match'] and comparison['accepted'] and diagnostics['estimator_calls']==len(genes)
                ref.append_jsonl(args.out/'lifecycle.jsonl',{'variant':label,'operation':operation,'passed':passed,
                    'comparison':comparison,'diagnostics':diagnostics,'before':before,'after':engine.snapshot()})
                if not passed:
                    failures.append({'variant':label,'case':operation})
        finally:
            engine.close()
            del engine
            gc.collect()
    for backend in ['rust','__UNKNOWN_BACKEND__']:
        status,value = outcome(lambda: HeadlessGoloco(args.repo,args.model_root,backend=backend,
                             library=args.out/'does_not_exist.so',process_limit_bytes=args.process_limit_bytes))
        if status['accepted']:
            status,_ = outcome(lambda: value.predict(frame,genes))
            value.close()
        passed = not status['accepted']
        ref.append_jsonl(args.out/'lifecycle.jsonl',{'case':'explicit_unavailable_'+backend,'outcome':status,'passed':passed})
        if not passed:
            failures.append({'case':'explicit_unavailable_'+backend})
    for backend in ['numpy','python_fast','rust']:
        budget = args.eviction_budget_bytes or eviction_budgets[backend]
        engine = HeadlessGoloco(args.repo,args.model_root,backend=backend,library=args.library,
                    max_cache_bytes=budget,process_limit_bytes=args.process_limit_bytes)
        try:
            for rep in range(2):
                result = engine.predict(frame,genes)
                comparison = frame_comparison(references['real'][1],result)
                assert comparison['exact_match'] and comparison['accepted']
            snap = engine.snapshot()
            passed = snap['evictions'] > 0 and snap['aggregate_cache_accounted_bytes'] <= budget
            ref.append_jsonl(args.out/'lifecycle.jsonl',{'case':'bounded_cache_eviction','variant':backend,
                'passed':passed,'snapshot':snap,'budget_bytes':budget})
            if not passed:
                failures.append({'variant':backend,'case':'bounded_cache_eviction'})
        finally:
            engine.close()
    # Identity changes are made solely to task-owned copies. Appending a byte to
    # a trusted pickle preserves pickle.load's result while changing its hash.
    copy_root = args.out/'source_identity_fixture'
    copy_models = copy_root/'L200_models'
    copy_models.mkdir(parents=True)
    gene = genes[0]
    for name in ['model_rd10_'+gene+'.pkl','feats_'+gene+'.csv']:
        shutil.copyfile(model_dir/name,copy_models/name)
    original.__globals__['_goloco_scope'] = [gene]
    single_expected = ref.original_output(original,frame,deadline)
    for backend in ['numpy','python_fast','rust']:
        for changed_name in ['model_rd10_'+gene+'.pkl','feats_'+gene+'.csv']:
            path = copy_models/changed_name
            shutil.copyfile(model_dir/changed_name,path)
            engine = HeadlessGoloco(args.repo,copy_root,backend=backend,library=args.library,
                        process_limit_bytes=args.process_limit_bytes)
            try:
                engine.predict(frame,[gene])
                with path.open('ab') as stream:
                    stream.write(b'\n')
                rejected,_ = outcome(lambda: engine.predict(frame,[gene]))
                passed = not rejected['accepted'] and rejected['type'].endswith('.SourceIdentityError')
                engine.invalidate(gene)
                actual = engine.predict(frame,[gene])
                comparison = frame_comparison(single_expected,actual)
                passed = passed and comparison['exact_match'] and comparison['accepted']
                ref.append_jsonl(args.out/'lifecycle.jsonl',{'case':'changed_source_reject_then_invalidate_reload',
                    'variant':backend,'changed_file':changed_name,'rejection':rejected,'comparison':comparison,'passed':passed})
                if not passed:
                    failures.append({'variant':backend,'case':'changed_source_'+changed_name})
            finally:
                engine.close()
                shutil.copyfile(model_dir/changed_name,path)
    if args.bundle:
        changed_name = 'model_rd10_'+gene+'.pkl'
        path = copy_models/changed_name
        engine = HeadlessGoloco(args.repo,copy_root,backend='rust',model_bundle=args.bundle,bundle_mode='mmap',
                    library=args.library,process_limit_bytes=args.process_limit_bytes)
        try:
            engine.predict(frame,[gene])
            with path.open('ab') as stream:
                stream.write(b'\n')
            rejected,_ = outcome(lambda: engine.predict(frame,[gene]))
            engine.invalidate(gene)
            rejected_after_invalidate,_ = outcome(lambda: engine.predict(frame,[gene]))
            passed = all(not item['accepted'] and item['type'].endswith('.SourceIdentityError')
                         for item in [rejected,rejected_after_invalidate])
            shutil.copyfile(model_dir/changed_name,path)
            engine.invalidate(gene)
            comparison = frame_comparison(single_expected,engine.predict(frame,[gene]))
            passed = passed and comparison['exact_match'] and comparison['accepted']
            ref.append_jsonl(args.out/'lifecycle.jsonl',{'case':'bundle_rejects_changed_source_even_after_invalidation',
                'passed':passed,'rejection':rejected,'after_invalidate':rejected_after_invalidate,'comparison':comparison})
            if not passed:
                failures.append({'case':'bundle_rejects_changed_source_even_after_invalidation'})
        finally:
            engine.close()
            shutil.copyfile(model_dir/changed_name,path)
    copy_repo = copy_root/'repo'
    (copy_repo/'data').mkdir(parents=True)
    for name in ['19q4_sum_stats.csv','19q4_gene_cats.csv']:
        shutil.copyfile(args.repo/'data'/name,copy_repo/'data'/name)
    changed_metadata = copy_repo/'data/19q4_sum_stats.csv'
    for backend in ['numpy','python_fast','rust']:
        engine = HeadlessGoloco(copy_repo,args.model_root,backend=backend,library=args.library,
                    process_limit_bytes=args.process_limit_bytes)
        try:
            engine.predict(frame,[gene])
            with changed_metadata.open('ab') as stream:
                stream.write(b'\n')
            rejected,_ = outcome(lambda: engine.predict(frame,[gene]))
            engine.invalidate()
            comparison = frame_comparison(single_expected,engine.predict(frame,[gene]))
            passed = (not rejected['accepted'] and rejected['type'].endswith('.SourceIdentityError')
                      and comparison['exact_match'] and comparison['accepted'])
            ref.append_jsonl(args.out/'lifecycle.jsonl',{'case':'changed_metadata_reject_then_invalidate_reload',
                'variant':backend,'passed':passed,'rejection':rejected,'comparison':comparison})
            if not passed:
                failures.append({'variant':backend,'case':'changed_metadata_reject_then_invalidate_reload'})
        finally:
            engine.close()
            shutil.copyfile(args.repo/'data/19q4_sum_stats.csv',changed_metadata)
    after_hashes = {str(path):ref.sha256(path) for path in paths}
    assert before_hashes == after_hashes
    ref.atomic_json(args.out/'source_integrity.json',{'unchanged':True,'files':before_hashes})
    ref.atomic_json(args.out/'summary.json',{'cases_per_variant':len(cases),'variants':len(variants),
        'oracle_contract_records':len(rows),'passed':sum(row['passed'] for row in rows),
        'failed':failures,'source_files_unchanged':len(paths),'status':'passed' if not failures else 'failed'})
    assert not failures, failures


if __name__ == '__main__':
    main()
