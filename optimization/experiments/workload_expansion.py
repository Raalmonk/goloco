#!/usr/bin/env python3
"""Optional one/32-experiment workloads after core acceptance; remote execution only."""
import argparse
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'reference'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['repo','model-root','input','targets','library','out']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--bundle',type=Path)
    p.add_argument('--deadline-epoch',type=float,required=True)
    p.add_argument('--process-limit-bytes',type=int,required=True)
    p.add_argument('--max-cache-bytes',type=int,default=16*1024**3)
    p.add_argument('--reps',type=int,default=5)
    p.add_argument('--pilot-targets',type=int,default=1024)
    p.add_argument('--pilot-target-file',type=Path,
                   help='Existing pilot target order; defaults to evidence/core_targets.json when present')
    p.add_argument('--attempt-full',action='store_true')
    p.add_argument('--matrix-audit-only',action='store_true')
    p.add_argument('--full-min-remaining-seconds',type=float,default=420)
    args=p.parse_args()
    assert Path('/content').is_dir()
    args.out.mkdir(parents=True,exist_ok=True)
    logging.disable(logging.CRITICAL)
    import numpy as np
    import pandas as pd
    from goloco_benchmark import (Memory,PreparedInput,atomic_json,append_jsonl,check_deadline,
        load_author,original_output,resolve_model_dir,feature_list,sha256)
    from goloco_experiment import correctness
    from goloco_reusable import HeadlessGoloco
    from goloco_reusable.native import FastForest
    from run_phase2 import summary
    deadline=time.monotonic()+args.deadline_epoch-time.time()
    canonical=json.loads(args.targets.read_text())
    source=pd.read_csv(args.input)
    original_experiments=[c for c in source.columns if c!='feature']
    model_dir=resolve_model_dir(args.model_root)
    if args.matrix_audit_only:
        from goloco_reusable.backend import PreparedInput as PackageInput
        candidate=PackageInput(source,original_experiments)
        widths={}
        pending=[]
        start=time.perf_counter()
        for gene in canonical:
            check_deadline(deadline)
            features=feature_list(model_dir,gene)
            selected=source.feature.isin(features)
            reference=source[selected][original_experiments].to_numpy().T
            positions=np.flatnonzero(selected.to_numpy())
            matrix,indices=candidate.select(features)
            assert np.array_equal(indices,positions)
            assert matrix.dtype==reference.dtype and matrix.shape==reference.shape
            assert np.array_equal(matrix,reference,equal_nan=True)
            width=int(matrix.shape[1])
            widths[width]=widths.get(width,0)+1
            pending.append(json.dumps({'gene':gene,'exact':True,'positions':positions.tolist(),
                'feature_ids':source.iloc[positions].feature.tolist(),'top_feature_file_entries':features,
                'dtype':str(matrix.dtype),'shape':list(matrix.shape),
                'sha256':hashlib.sha256(matrix.tobytes(order='C')).hexdigest()})+'\n')
            if len(pending)>=256 or gene==canonical[-1]:
                with (args.out/'matrix_audit.jsonl').open('a') as stream:
                    stream.writelines(pending)
                    stream.flush()
                    os.fsync(stream.fileno())
                pending.clear()
        atomic_json(args.out/'summary.json',{'target_count':len(canonical),'all_matrices_exact':True,
            'feature_width_counts':widths,'wall_seconds':time.perf_counter()-start,
            'classification':'validation only; no models loaded or inference performed'})
        return
    pilot_path=args.pilot_target_file
    if pilot_path is None and (ROOT/'evidence/core_targets.json').is_file():
        pilot_path=ROOT/'evidence/core_targets.json'
    pilot_source=json.loads(pilot_path.read_text()) if pilot_path is not None else canonical
    assert isinstance(pilot_source,list) and all(isinstance(gene,str) for gene in pilot_source)
    assert len(pilot_source)==len(set(pilot_source)), 'Pilot target list contains duplicates'
    assert set(pilot_source).issubset(set(canonical)), 'Pilot target is outside the supported canonical set'
    assert 1<=args.pilot_targets<=len(pilot_source), 'Requested pilot count exceeds the existing target list'
    pilot_genes=pilot_source[:args.pilot_targets]
    assert len(pilot_genes)==args.pilot_targets
    pilot_identity={'target_count':len(pilot_genes),'targets':pilot_genes,
        'target_list_sha256':hashlib.sha256(json.dumps(pilot_genes,separators=(',',':')).encode()).hexdigest(),
        'source_file':str(pilot_path) if pilot_path is not None else str(args.targets),
        'source_file_sha256':sha256(pilot_path if pilot_path is not None else args.targets),
        'selection':'prefix of existing pilot target order' if pilot_path is not None else 'canonical order fallback; pilot evidence file absent'}
    one=source[['feature',original_experiments[0]]].copy()
    synthetic=pd.DataFrame({'feature':source.feature.copy()})
    row=np.arange(len(source),dtype=np.float64)
    for index in range(32):
        # Deterministic, distinct performance-only columns, never biological claims.
        base=source[original_experiments[index%len(original_experiments)]].to_numpy()
        perturbation=(index+1)*.0001 + np.sin((row+1)*(index+1)*.013)*.005
        synthetic['SYNTH_PERFORMANCE_%02d'%(index+1)]=base+perturbation
    cases={'one_original_experiment':one,'32_synthetic_performance_experiments':synthetic}
    inputs={}
    for name,frame in cases.items():
        path=args.out/(name+'.csv')
        frame.to_csv(path,index=False)
        inputs[name]=path
    atomic_json(args.out/'workload_manifest.json',{'source_input_sha256':sha256(args.input),'seed':20261005,
        'pilot':pilot_identity,
        'synthetic_purpose':'performance scaling only; not biological validation',
        'synthetic_formula':'original[column i mod 3] + (i+1)*0.0001 + sin((row+1)*(i+1)*0.013)*0.005',
        'cases':{name:{'input':str(path),'sha256':sha256(path),'columns':cases[name].columns.tolist()}
                 for name,path in inputs.items()},'deadline_epoch':args.deadline_epoch})
    pilot_elapsed=None
    for scope_name,genes in [('pilot',pilot_genes),('full',canonical)]:
        if scope_name=='full':
            estimated=(pilot_elapsed or 0)*len(canonical)/len(pilot_genes)*1.3+30
            required=max(args.full_min_remaining_seconds,estimated)
            if not args.attempt_full or args.deadline_epoch-time.time()<required:
                atomic_json(args.out/'full_status.json',{'state':'not_started','remaining_seconds':args.deadline_epoch-time.time(),
                    'estimated_required_seconds':required,'full_requested':args.attempt_full,
                    'completed_pilot_preserved':True})
                break
        tier=args.out/scope_name
        tier.mkdir()
        tier_start=time.perf_counter()
        module,original=load_author(args.repo,model_dir,genes,tier,neutral_progress=True)
        counted=[0]
        method=module.infer.infer_gene
        def author_counted(instance,gene):
            check_deadline(deadline)
            counted[0]+=1
            return method(instance,gene)
        module.infer.infer_gene=author_counted
        expected={}
        for name,input_path in inputs.items():
            check_deadline(deadline)
            counted[0]=0
            with Memory() as memory:
                start,cpu=time.perf_counter(),time.process_time()
                frame=pd.read_csv(input_path)
                output=original_output(original,frame,deadline)
                output.to_csv(tier/(name+'_oracle.csv'),index=True)
                seconds,cpu_seconds=time.perf_counter()-start,time.process_time()-cpu
            assert counted[0]==len(genes)
            assert memory.peak<=args.process_limit_bytes
            output.to_pickle(tier/(name+'_oracle.pkl'))
            expected[name]=output
            append_jsonl(tier/'original_timings.jsonl',{'arm':'A','workload':name,'wall_seconds':seconds,
                'cpu_seconds':cpu_seconds,'target_count':len(genes),'estimator_calls':counted[0],
                'memory':memory.result(),'csv_sha256':sha256(tier/(name+'_oracle.csv'))})
        # One source/model residency. Switching predictors retains identical
        # originals and uses the same verified package selection/output code.
        start=time.perf_counter()
        engine=HeadlessGoloco(args.repo,args.model_root,backend='rust',model_bundle=args.bundle,
            bundle_mode='mmap',library=args.library,max_cache_bytes=args.max_cache_bytes,
            process_limit_bytes=args.process_limit_bytes)
        engine.prepare(genes)
        assert set(engine._entries)==set(genes)
        rust={gene:entry['predictor'] for gene,entry in engine._entries.items()}
        b3={gene:FastForest(entry['model']) for gene,entry in engine._entries.items()}
        extra_wrapper_charge=len(b3)*2048
        assert engine.snapshot()['aggregate_cache_accounted_bytes']+extra_wrapper_charge<=args.max_cache_bytes
        atomic_json(tier/'shared_setup.json',{'seconds':time.perf_counter()-start,'snapshot':engine.snapshot(),
            'memory_class':'paired common source models plus Rust and B3 adapters',
            'extra_B3_wrappers':len(b3),'extra_B3_wrapper_accounted_bytes':extra_wrapper_charge,
            'no_extra_tree_buffers_for_B3':True})
        rows=[]
        try:
            initial=int(np.random.RandomState(20261005).randint(2))
            for name,input_path in inputs.items():
                sink=tier/'same_output_sink.csv'
                for rep in range(-1,args.reps):
                    order=['B3','C0'] if (rep+initial)%2==0 else ['C0','B3']
                    for arm in order:
                        check_deadline(deadline)
                        engine.backend='python_fast' if arm=='B3' else 'rust'
                        for gene,entry in engine._entries.items():
                            entry['predictor']=(b3 if arm=='B3' else rust)[gene]
                        diagnostics={}
                        with Memory() as memory:
                            start,cpu=time.perf_counter(),time.process_time()
                            frame=pd.read_csv(input_path)
                            output=engine.predict(frame,genes,diagnostics=diagnostics)
                            output.to_csv(sink,index=True)
                            seconds,cpu_seconds=time.perf_counter()-start,time.process_time()-cpu
                        experiments=[c for c in frame.columns if c!='feature']
                        checks=correctness(expected[name],output,genes,experiments)
                        checks['serialized_csv_bytes_exact']=sha256(sink)==sha256(tier/(name+'_oracle.csv'))
                        assert checks['serialized_csv_bytes_exact'] and diagnostics['estimator_calls']==len(genes)
                        assert memory.peak<=args.process_limit_bytes and set(engine._entries)==set(genes)
                        item={'arm':arm,'workload':name,'repetition':rep,'wall_seconds':seconds,
                            'cpu_seconds':cpu_seconds,'target_count':len(genes),'experiment_count':len(experiments),
                            'diagnostics':diagnostics,'correctness':checks,'memory':memory.result()}
                        append_jsonl(tier/('prevalidation.jsonl' if rep<0 else 'timings.jsonl'),item)
                        if rep>=0:
                            rows.append(item)
            atomic_json(tier/'summary.json',{name:summary([row for row in rows if row['workload']==name]) for name in cases})
            atomic_json(tier/'status.json',{'state':'complete','epoch':time.time(),'samples':len(rows)})
        finally:
            b3.clear()
            rust.clear()
            engine.close()
            # The final loop entry may own a view into the entire mapped
            # bundle; release it before opening the next scope's service.
            entry=None
            engine=None
            gc.collect()
        if scope_name=='pilot':
            pilot_elapsed=time.perf_counter()-tier_start


if __name__=='__main__':
    main()
