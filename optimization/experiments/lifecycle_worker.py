#!/usr/bin/env python3
"""One fresh-process lifecycle worker; first inference is the timed request."""
# Keep initial imports standard-library only. Parent starts its timer before exec.
import argparse
import json
import logging
from pathlib import Path
import resource
import shutil
import sys
import time


def main():
    entry = time.perf_counter()
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    assert Path('/content').is_dir()
    assert time.time() < config['deadline_epoch']
    logging.disable(logging.CRITICAL)
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / 'reference'))
    import numpy as np
    import pandas as pd
    from goloco_benchmark import (Memory, ModelCache, atomic_json, append_jsonl, check_deadline,
                                  predict_optimized, resolve_model_dir, sha256)
    out, repo = Path(config['out']), Path(config['repo'])
    genes = config['targets']
    deadline = time.monotonic() + config['deadline_epoch'] - time.time()
    imports_done = time.perf_counter()
    variant = config['variant']
    setup_start = time.perf_counter()
    if variant == 'historical_c0':
        from rust_candidate import RustLibrary, RustForest, CacheView, export_size, aggregate_rss
        model_dir = resolve_model_dir(Path(config['model_root']))
        library = RustLibrary(Path(config['library']))
        cache = ModelCache(repo, model_dir, genes, deadline=deadline, max_bytes=config['max_cache_bytes'],
                           process_limit_bytes=config['process_limit_bytes'], eager=True)
        metadata_bytes = int(cache.stats.memory_usage(index=True, deep=True).sum()+cache.categories.memory_usage(index=True, deep=True).sum())
        metadata_bytes += sum(sys.getsizeof(k)+sys.getsizeof(v)+sum(sys.getsizeof(x) for x in v) for k,v in cache.features.items())
        derived = sum(export_size(model) for model in cache.models.values())
        assert cache.retained_bytes + metadata_bytes + derived <= config['max_cache_bytes']
        assert set(cache.models) == set(genes)
        assert aggregate_rss() + derived <= config['process_limit_bytes']
        adapters = {}
        for gene in genes:
            check_deadline(deadline)
            adapters[gene] = RustForest(cache.models[gene], library)
            assert aggregate_rss() <= config['process_limit_bytes']
        view = CacheView(cache, adapters)
        def predict(frame, diagnostics):
            result = predict_optimized(frame, genes, repo, model_dir, True, view, deadline, diagnostics)
            diagnostics['ffi_calls'] = diagnostics['estimator_calls']
            return result
        def snapshot():
            return dict(cache=cache.snapshot(), derived_bytes=derived,
                        aggregate_accounted_bytes=cache.retained_bytes+metadata_bytes+derived)
        engine = None
    else:
        from goloco_reusable import HeadlessGoloco
        backend = 'python_fast' if variant == 'source_python_fast' else 'rust'
        bundle = config['bundle'] if variant in ['packed_load', 'packed_mmap'] else None
        if variant in ['packed_load', 'packed_mmap']:
            assert bundle is not None
        engine = HeadlessGoloco(repo, Path(config['model_root']), backend=backend,
                    model_bundle=Path(bundle) if bundle else None,
                    bundle_mode='load' if variant == 'packed_load' else 'mmap',
                    library=Path(config['library']), max_cache_bytes=config['max_cache_bytes'],
                    process_limit_bytes=config['process_limit_bytes'], chunk_size=config['chunk_size'])
        def predict(frame, diagnostics):
            return engine.predict(frame, genes, diagnostics=diagnostics)
        snapshot = engine.snapshot
    setup_seconds = time.perf_counter()-setup_start
    sink = Path(config['sink'])
    counters = {}
    first_usage = resource.getrusage(resource.RUSAGE_SELF)
    start, cpu = time.perf_counter(), time.process_time()
    frame = pd.read_csv(config['input'])
    first = predict(frame, counters)
    first.to_csv(sink, index=True)
    end = time.perf_counter()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # This receipt is emitted before hashing, pickling, oracle reading or validation.
    print(json.dumps({'event': 'first_output_closed', 'variant': variant,
        'worker_entry_to_close_seconds': end-entry, 'imports_seconds': imports_done-entry,
        'setup_seconds': setup_seconds, 'request_seconds': end-start,
        'request_cpu_seconds': time.process_time()-cpu, 'counters': counters,
        'minor_faults_process_to_output': usage.ru_minflt, 'major_faults_process_to_output': usage.ru_majflt,
        'minor_faults_request': usage.ru_minflt-first_usage.ru_minflt,
        'major_faults_request': usage.ru_majflt-first_usage.ru_majflt,
        'high_water_rss_bytes': usage.ru_maxrss*1024}), flush=True)
    assert counters['estimator_calls'] == len(genes)
    first.to_pickle(out/'first_output.pkl')
    shutil.copyfile(sink, out/'first_output.csv')
    first_hash = sha256(sink)
    atomic_json(out/'first_snapshot.json', snapshot())
    for rep in range(config['warm_reps']):
        check_deadline(deadline)
        diagnostics, before = {}, resource.getrusage(resource.RUSAGE_SELF)
        before_cache = snapshot()
        with Memory() as memory:
            start, cpu = time.perf_counter(), time.process_time()
            frame = pd.read_csv(config['input'])
            result = predict(frame, diagnostics)
            result.to_csv(sink, index=True)
            seconds, cpu_seconds = time.perf_counter()-start, time.process_time()-cpu
        after = resource.getrusage(resource.RUSAGE_SELF)
        # Full oracle validation is external to this worker and performed after timing.
        pd.testing.assert_frame_equal(first, result, check_exact=True)
        assert sha256(sink) == first_hash
        assert diagnostics['estimator_calls'] == len(genes)
        assert memory.peak <= config['process_limit_bytes']
        append_jsonl(out/'resident_timings.jsonl', {'arm': variant, 'repetition': rep, 'wall_seconds': seconds,
            'cpu_seconds': cpu_seconds, 'target_count': len(genes), 'counters': diagnostics,
            'memory': memory.result(), 'memory_class': 'standalone service',
            'minor_faults': after.ru_minflt-before.ru_minflt, 'major_faults': after.ru_majflt-before.ru_majflt,
            'cache_before': before_cache, 'cache_after': snapshot(), 'exact_to_first_request': True,
            'csv_sha256': first_hash})
    if engine is not None:
        engine.close()
        atomic_json(out/'closed_snapshot.json', engine.snapshot())
    atomic_json(out/'status.json', {'state': 'complete', 'epoch': time.time()})


if __name__ == '__main__':
    main()
