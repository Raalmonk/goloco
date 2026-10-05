#!/usr/bin/env python3
"""Independent diagnostic profile; never mixed into headline request samples."""
import argparse
from collections import defaultdict
import ctypes
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'reference'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['repo','model-root','input','targets','library','oracle','out']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--deadline-epoch',type=float,required=True)
    p.add_argument('--process-limit-bytes',type=int,required=True)
    p.add_argument('--max-cache-bytes',type=int,default=16*1024**3)
    p.add_argument('--limit-targets',type=int)
    args=p.parse_args()
    assert Path('/content').is_dir()
    args.out.mkdir(parents=True,exist_ok=True)
    import numpy as np
    import pandas as pd
    from goloco_benchmark import (Memory,ModelCache,PreparedInput,atomic_json,append_jsonl,
        check_deadline,resolve_model_dir,output_table,sha256)
    from goloco_experiment import correctness
    from native_candidate import FastForest
    from rust_candidate import RustLibrary,RustForest,export_size,aggregate_rss
    deadline=time.monotonic()+args.deadline_epoch-time.time()
    genes=json.loads(args.targets.read_text())
    if args.limit_targets:
        genes=genes[:args.limit_targets]
    model_dir=resolve_model_dir(args.model_root)
    expected=pd.read_pickle(args.oracle/'oracle_real.pkl').iloc[:len(genes)].copy()
    library=RustLibrary(args.library)
    profile=library.handle.goloco_predict_forest_profile
    profile.argtypes=library.predict.argtypes+[ctypes.c_void_p]
    profile.restype=ctypes.c_int32
    with Memory() as memory:
        start=time.perf_counter()
        cache=ModelCache(args.repo,model_dir,genes,deadline=deadline,max_bytes=args.max_cache_bytes,
                         process_limit_bytes=args.process_limit_bytes,eager=True)
        preload=time.perf_counter()-start
    derived=sum(export_size(model) for model in cache.models.values())
    metadata_bytes=int(cache.stats.memory_usage(index=True,deep=True).sum()+cache.categories.memory_usage(index=True,deep=True).sum())
    metadata_bytes+=sum(sys.getsizeof(k)+sys.getsizeof(v)+sum(sys.getsizeof(x) for x in v) for k,v in cache.features.items())
    assert cache.retained_bytes+metadata_bytes+derived<=args.max_cache_bytes
    assert aggregate_rss()+derived<=args.process_limit_bytes
    assert set(cache.models)==set(genes)
    start=time.perf_counter()
    native={}
    for gene in genes:
        check_deadline(deadline)
        native[gene]=RustForest(cache.models[gene],library)
    export=time.perf_counter()-start
    atomic_json(args.out/'setup.json',{'preload_seconds':preload,'export_seconds':export,
        'source_accounted_bytes':cache.retained_bytes,'derived_bytes':derived,
        'memory':memory.result(),'target_count':len(genes)})
    for arm in ['B3','C0']:
        stages=defaultdict(float)
        counts=defaultdict(int)
        t=time.perf_counter()
        frame=pd.read_csv(args.input)
        stages['csv_parse']+=time.perf_counter()-t
        experiments=[c for c in frame.columns if c!='feature']
        t=time.perf_counter()
        prepared=PreparedInput(frame,experiments)
        predictions=np.zeros((len(genes),len(experiments)))
        zscores=np.zeros_like(predictions)
        averages,stds=[],[]
        stages['input_index_and_output_allocation']+=time.perf_counter()-t
        for i,gene in enumerate(genes):
            check_deadline(deadline)
            t=time.perf_counter()
            model=cache.get_model(gene)
            features=cache.get_features(gene)
            average,std=cache.get_statistics(gene)
            stages['cache_and_metadata_lookup']+=time.perf_counter()-t
            t=time.perf_counter()
            matrix,positions=prepared.select(features)
            stages['feature_selection_and_data_preparation']+=time.perf_counter()-t
            counts['selected_matrix_bytes']+=matrix.nbytes
            t=time.perf_counter()
            checked=model._validate_X_predict(matrix)
            stages['legacy_validation_and_float32_cast']+=time.perf_counter()-t
            counts['validator_calls']+=1
            if not np.shares_memory(checked,matrix):
                counts['validator_copied_bytes']+=checked.nbytes
                counts['validator_copies']+=1
            t=time.perf_counter()
            result=np.zeros(checked.shape[0],dtype=np.float64) if arm=='B3' else np.empty(checked.shape[0],dtype=np.float64)
            stages['prediction_output_allocation']+=time.perf_counter()-t
            counts['prediction_allocations']+=1
            counts['prediction_allocated_bytes']+=result.nbytes
            if arm=='B3':
                t=time.perf_counter()
                for tree in model.estimators_:
                    result+=tree.tree_.predict(checked)[:,0]
                result/=len(model.estimators_)
                stages['tree_native_predict_python_iteration_and_ordered_sum']+=time.perf_counter()-t
                counts['tree_calls']+=len(model.estimators_)
            else:
                adapter=native[gene]
                native_seconds=ctypes.c_double()
                t=time.perf_counter()
                status=profile(*adapter.pointers,adapter.n_nodes,adapter.n_trees,
                    checked.ctypes.data,checked.shape[0],checked.shape[1],checked.strides[0],checked.strides[1],
                    result.ctypes.data,ctypes.byref(native_seconds))
                ffi_seconds=time.perf_counter()-t
                assert status==0
                stages['ffi_interval_including_native_kernel']+=ffi_seconds
                stages['native_bounds_traversal_and_ordered_sum']+=native_seconds.value
                stages['estimated_ctypes_boundary_overhead']+=ffi_seconds-native_seconds.value
                counts['ffi_calls']+=1
            counts['estimator_calls']+=1
            t=time.perf_counter()
            predictions[i]=result
            zscores[i]=(result-average)/std
            averages.append(average)
            stds.append(std)
            stages['zscore_and_output_array_fill']+=time.perf_counter()-t
        t=time.perf_counter()
        output=output_table(genes,cache.categories,predictions,zscores,averages,stds,experiments)
        stages['output_table_construction']+=time.perf_counter()-t
        sink=args.out/'same_output_sink.csv'
        t=time.perf_counter()
        output.to_csv(sink,index=True)
        stages['csv_serialization_close_no_fsync']+=time.perf_counter()-t
        checks=correctness(expected,output,genes,experiments)
        expected.to_csv(args.out/'reference_subset.csv',index=True)
        checks['serialized_csv_bytes_exact']=sha256(sink)==sha256(args.out/'reference_subset.csv')
        assert checks['serialized_csv_bytes_exact']
        assert counts['estimator_calls']==len(genes)
        record={'arm':arm,'target_count':len(genes),'stages_seconds':dict(stages),'counts':dict(counts),
            'correctness':checks,'classification':'instrumented diagnostic; not headline timing',
            'native_definition':'Rust Instant interval includes native argument/bounds checks, traversal and accumulation',
            'ffi_definition':'Python wall interval includes argument marshaling plus native; difference is estimated boundary cost',
            'stage_overlap':'ffi_interval includes native and estimated boundary; do not sum nested stages',
            'memory_class':'paired source models and Rust adapters retained'}
        append_jsonl(args.out/'profiles.jsonl',record)
    atomic_json(args.out/'status.json',{'state':'complete','epoch':time.time()})


if __name__=='__main__':
    main()
