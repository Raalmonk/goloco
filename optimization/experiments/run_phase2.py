#!/usr/bin/env python3
"""Independent Phase 2 historical replication and parent-observed lifecycle trials.

All inference must run on the dedicated Colab VM. CSV boundaries use pandas
close/flush, without fsync, identically in every arm. JSON evidence is durable.
"""
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import queue
import resource
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reference'))


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['original', 'paired', 'lifecycle'])
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--model-root', type=Path, required=True)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--targets', type=Path, required=True)
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--oracle', type=Path)
    p.add_argument('--bundle', type=Path)
    p.add_argument('--deadline-epoch', type=float, required=True)
    p.add_argument('--reps', type=int, default=5)
    p.add_argument('--startup-reps', type=int, default=3)
    p.add_argument('--warm-reps', type=int, default=5)
    p.add_argument('--variants', default='historical_c0,source_python_fast,source_rust,packed_load,packed_mmap')
    p.add_argument('--max-cache-bytes', type=int, default=16 * 1024 ** 3)
    p.add_argument('--process-limit-bytes', type=int, required=True)
    p.add_argument('--chunk-size', type=int, default=1)
    return p.parse_args()


def targets(path):
    value = json.loads(path.read_text())
    if isinstance(value, dict):
        value = value.get('targets', value.get('genes'))
    assert isinstance(value, list) and len(value) == len(set(value))
    return value


def faults():
    value = resource.getrusage(resource.RUSAGE_SELF)
    return {'minor': value.ru_minflt, 'major': value.ru_majflt}


def subtract(a, b):
    return {key: a[key] - b[key] for key in a}


def summary(records, key='arm'):
    import numpy as np
    result = {}
    for arm in sorted(set(x[key] for x in records)):
        numbers = np.array([x['wall_seconds'] for x in records if x[key] == arm])
        result[arm] = {'n': len(numbers), 'median_seconds': float(np.median(numbers)),
                       'minimum_seconds': float(numbers.min()), 'maximum_seconds': float(numbers.max()),
                       'iqr_seconds': float(np.percentile(numbers, 75) - np.percentile(numbers, 25))}
    return result


def historical(args):
    import numpy as np
    import pandas as pd
    import psutil
    from goloco_benchmark import (Memory, ModelCache, Timer, atomic_json, append_jsonl,
        check_deadline, load_author, original_output, predict_optimized, resolve_model_dir, sha256)
    from goloco_experiment import correctness
    from native_candidate import FastForest
    from rust_candidate import RustLibrary, RustForest, CacheView, export_size, aggregate_rss
    genes = targets(args.targets)
    model_dir = resolve_model_dir(args.model_root)
    deadline = time.monotonic() + args.deadline_epoch - time.time()
    experiments = list(pd.read_csv(args.input, nrows=0).columns)[1:]
    protocol = {'repo': str(args.repo), 'model_dir': str(model_dir), 'input': str(args.input),
                'targets': genes, 'experiments': experiments, 'csv_index': True,
                'aggregate_memory_limit_bytes': args.process_limit_bytes, 'cache_budget_bytes': args.max_cache_bytes,
                'boundary': 'CSV parse through closed CSV output; no fsync; UI/progress excluded',
                'seed': 20261005, 'input_sha256': sha256(args.input), 'library_sha256': sha256(args.library),
                'deadline_epoch': args.deadline_epoch, 'mode': args.mode}
    atomic_json(args.out / 'protocol.json', protocol)
    records = []
    sink = args.out / 'same_output_sink.csv'
    if args.mode == 'original':
        module, original = load_author(args.repo, model_dir, genes, args.out, neutral_progress=True)
        count = [0]
        method = module.infer.infer_gene
        def counted(instance, gene):
            check_deadline(deadline)
            count[0] += 1
            return method(instance, gene)
        module.infer.infer_gene = counted
        expected = None
        for rep in range(args.reps):
            check_deadline(deadline)
            count[0] = 0
            before = faults()
            with Memory() as memory:
                start, cpu = time.perf_counter(), time.process_time()
                frame = pd.read_csv(args.input)
                result = original_output(original, frame, deadline)
                result.to_csv(sink, index=True)
                elapsed, cpu_seconds = time.perf_counter()-start, time.process_time()-cpu
            after = faults()
            assert count[0] == len(genes)
            assert memory.peak <= args.process_limit_bytes
            if expected is None:
                expected = result.copy(deep=True)
                expected.to_pickle(args.out / 'oracle_real.pkl')
                shutil.copyfile(sink, args.out / 'oracle_real.csv')
            checks = correctness(expected, result, genes, experiments)
            checks['serialized_csv_bytes_exact'] = sha256(sink) == sha256(args.out / 'oracle_real.csv')
            assert checks['serialized_csv_bytes_exact']
            row = {'arm': 'A', 'repetition': rep, 'wall_seconds': elapsed, 'cpu_seconds': cpu_seconds,
                   'estimator_calls': count[0], 'target_count': len(genes), 'memory': memory.result(),
                   'page_faults': subtract(after, before), 'correctness': checks,
                   'csv_sha256': sha256(sink), 'state': 'application_cold' if rep == 0 else 'repeat_no_application_cache'}
            records.append(row)
            append_jsonl(args.out / 'timings.jsonl', row)
            print(json.dumps({'event': 'completed', 'arm': 'A', 'rep': rep, 'seconds': elapsed}), flush=True)
    else:
        assert args.oracle is not None
        expected = pd.read_pickle(args.oracle / 'oracle_real.pkl')
        oracle_hash = sha256(args.oracle / 'oracle_real.csv')
        build_timer = Timer()
        with Memory() as memory:
            start, cpu = time.perf_counter(), time.process_time()
            cache = ModelCache(args.repo, model_dir, genes, build_timer, deadline,
                               max_bytes=args.max_cache_bytes, process_limit_bytes=args.process_limit_bytes, eager=True)
            seconds, cpu_seconds = time.perf_counter()-start, time.process_time()-cpu
        assert set(cache.models) == set(genes)
        atomic_json(args.out / 'preload.json', {'wall_seconds': seconds, 'cpu_seconds': cpu_seconds,
                   'memory': memory.result(), 'cache': cache.snapshot(), 'stages_seconds': build_timer.values})
        metadata_bytes = int(cache.stats.memory_usage(index=True, deep=True).sum() + cache.categories.memory_usage(index=True, deep=True).sum())
        metadata_bytes += sum(sys.getsizeof(k)+sys.getsizeof(v)+sum(sys.getsizeof(x) for x in v) for k,v in cache.features.items())
        derived = sum(export_size(model) for model in cache.models.values())
        assert cache.retained_bytes + metadata_bytes + derived <= args.max_cache_bytes
        assert aggregate_rss() + derived <= args.process_limit_bytes
        library = RustLibrary(args.library)
        with Memory() as memory:
            start, cpu = time.perf_counter(), time.process_time()
            b3 = {gene: FastForest(cache.models[gene]) for gene in genes}
            native = {}
            for gene in genes:
                check_deadline(deadline)
                native[gene] = RustForest(cache.models[gene], library)
                assert aggregate_rss() <= args.process_limit_bytes
            seconds, cpu_seconds = time.perf_counter()-start, time.process_time()-cpu
        atomic_json(args.out / 'export.json', {'wall_seconds': seconds, 'cpu_seconds': cpu_seconds,
                   'derived_buffer_bytes': derived, 'source_models_accounted_bytes': cache.retained_bytes,
                   'combined_cache_accounted_bytes': cache.retained_bytes + metadata_bytes + derived,
                   'memory': memory.result(), 'memory_class': 'paired shared source models with both adapters resident'})
        views = {'B3': CacheView(cache, b3), 'C0': CacheView(cache, native)}
        def run(arm, rep, measured):
            check_deadline(deadline)
            counter, before_cache, before_faults = {}, cache.snapshot(), faults()
            with Memory() as memory:
                start, cpu = time.perf_counter(), time.process_time()
                frame = pd.read_csv(args.input)
                result = predict_optimized(frame, genes, args.repo, model_dir, True, views[arm], deadline, counter)
                result.to_csv(sink, index=True)
                elapsed, cpu_seconds = time.perf_counter()-start, time.process_time()-cpu
            after_faults = faults()
            checks = correctness(expected, result, genes, experiments)
            checks['serialized_csv_bytes_exact'] = sha256(sink) == oracle_hash
            assert checks['serialized_csv_bytes_exact'] and counter['estimator_calls'] == len(genes)
            after_cache = cache.snapshot()
            assert before_cache['misses'] == after_cache['misses'] and before_cache['evictions'] == after_cache['evictions']
            assert memory.peak <= args.process_limit_bytes
            row = {'arm': arm, 'repetition': rep, 'measured': measured, 'wall_seconds': elapsed,
                   'cpu_seconds': cpu_seconds, 'estimator_calls': counter['estimator_calls'],
                   'ffi_calls': len(genes) if arm == 'C0' else 0, 'target_count': len(genes),
                   'correctness': checks, 'csv_sha256': sha256(sink), 'memory': memory.result(),
                   'memory_class': 'paired shared source models and both adapters resident',
                   'page_faults': subtract(after_faults, before_faults), 'cache_before': before_cache, 'cache_after': after_cache}
            append_jsonl(args.out / ('timings.jsonl' if measured else 'prevalidation.jsonl'), row)
            if measured:
                records.append(row)
            print(json.dumps({'event': 'completed', 'arm': arm, 'rep': rep, 'measured': measured, 'seconds': elapsed}), flush=True)
        for arm in ['B3', 'C0']:
            run(arm, -1, False)
        initial = int(np.random.RandomState(20261005).randint(2))
        orders = []
        for rep in range(args.reps):
            order = ['B3', 'C0'] if (rep + initial) % 2 == 0 else ['C0', 'B3']
            orders.append(order)
            atomic_json(args.out / 'arm_order.json', orders)
            for arm in order:
                run(arm, rep, True)
    atomic_json(args.out / 'summary.json', summary(records))
    atomic_json(args.out / 'status.json', {'state': 'complete', 'samples': len(records), 'epoch': time.time()})


def lifecycle(args):
    import numpy as np
    import pandas as pd
    import psutil
    from goloco_benchmark import atomic_json, append_jsonl, sha256
    from goloco_experiment import correctness
    assert args.oracle is not None
    variants = args.variants.split(',')
    base = list(np.random.RandomState(20261005).permutation(variants))
    genes = targets(args.targets)
    records, orders = [], []
    for trial in range(args.startup_reps):
        order = base[trial % len(base):] + base[:trial % len(base)]
        orders.append(order)
        atomic_json(args.out / 'arm_order.json', orders)
        for variant in order:
            assert time.time() < args.deadline_epoch
            trial_dir = args.out / (variant + '_' + str(trial))
            trial_dir.mkdir()
            config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
            config.update(variant=variant, out=str(trial_dir), sink=str(args.out / 'same_output_sink.csv'),
                          targets=genes, warm_reps=args.warm_reps if trial == 0 else 0)
            atomic_json(trial_dir / 'config.json', config)
            command = [sys.executable, str(Path(__file__).with_name('lifecycle_worker.py')), '--config', str(trial_dir/'config.json')]
            events = queue.Queue()
            start = time.perf_counter()
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
            peak = [0]
            stopped = threading.Event()
            def monitor():
                proc = psutil.Process(process.pid)
                while not stopped.wait(.01):
                    try:
                        rss = proc.memory_info().rss + sum(child.memory_info().rss for child in proc.children(recursive=True))
                        peak[0] = max(peak[0], rss)
                        if rss > args.process_limit_bytes:
                            process.kill()
                            events.put(('failure', 'aggregate memory ceiling exceeded', time.perf_counter()))
                            return
                    except psutil.Error:
                        return
            def read_stdout():
                for line in process.stdout:
                    events.put(('stdout', line, time.perf_counter()))
                events.put(('eof', '', time.perf_counter()))
            errors = []
            def read_stderr():
                for line in process.stderr:
                    errors.append(line)
            monitors = [threading.Thread(target=fn, daemon=True) for fn in [monitor, read_stdout, read_stderr]]
            for thread in monitors:
                thread.start()
            first = None
            transcript = []
            try:
                while True:
                    remaining = args.deadline_epoch-time.time()
                    if remaining <= 0:
                        process.kill()
                        raise TimeoutError('global deadline reached')
                    try:
                        kind, line, observed = events.get(timeout=min(1, remaining))
                    except queue.Empty:
                        continue
                    if kind == 'failure':
                        raise MemoryError(line)
                    if kind == 'eof':
                        break
                    transcript.append(line)
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if event.get('event') == 'first_output_closed':
                        assert first is None
                        first = {'arm': variant, 'trial': trial, 'wall_seconds': observed-start,
                                 'boundary': 'parent immediately before Popen through receipt immediately after first CSV close; no fsync',
                                 'worker_event': event, 'sampled_peak_rss_to_first_output_bytes': peak[0],
                                 'memory_class': 'standalone worker including descendants; parent excluded',
                                 'os_cache_policy': 'normal cache state, no cache flush or inference prevalidation'}
                        atomic_json(trial_dir / 'first_output_timing.json', first)
                returncode = process.wait(timeout=max(1, args.deadline_epoch-time.time()))
            finally:
                stopped.set()
                if process.poll() is None:
                    process.kill()
                    process.wait()
                for thread in monitors:
                    thread.join(timeout=2)
                (trial_dir / 'stdout.txt').write_text(''.join(transcript))
                (trial_dir / 'stderr.txt').write_text(''.join(errors))
            assert returncode == 0, 'worker failure: ' + ''.join(errors)[-4000:]
            assert first is not None
            # Reference and result are loaded only AFTER the parent first-output boundary.
            expected = pd.read_pickle(args.oracle / 'oracle_real.pkl')
            actual = pd.read_pickle(trial_dir / 'first_output.pkl')
            experiments = [x for x in pd.read_csv(args.input, nrows=0).columns if x != 'feature']
            checks = correctness(expected, actual, genes, experiments)
            checks['serialized_csv_bytes_exact'] = sha256(trial_dir/'first_output.csv') == sha256(args.oracle/'oracle_real.csv')
            assert checks['serialized_csv_bytes_exact']
            first.update(correctness=checks, csv_sha256=sha256(trial_dir/'first_output.csv'),
                         sampled_peak_rss_whole_worker_bytes=peak[0], returncode=returncode)
            append_jsonl(args.out / 'timings.jsonl', first)
            atomic_json(trial_dir / 'first_output_timing.json', first)
            records.append(first)
            print(json.dumps({'event': 'startup_complete', 'arm': variant, 'trial': trial, 'seconds': first['wall_seconds']}), flush=True)
    atomic_json(args.out / 'summary.json', summary(records))
    atomic_json(args.out / 'status.json', {'state': 'complete', 'samples': len(records), 'epoch': time.time()})


def main():
    args = arguments()
    assert Path('/content').is_dir(), 'Execution is restricted to the dedicated Colab VM'
    logging.disable(logging.CRITICAL)
    args.out.mkdir(parents=True, exist_ok=True)
    assert not (args.out / 'timings.jsonl').exists(), 'Do not overwrite completed measurements'
    if args.mode == 'lifecycle':
        lifecycle(args)
    else:
        historical(args)


if __name__ == '__main__':
    main()
