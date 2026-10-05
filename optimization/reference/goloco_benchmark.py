#!/usr/bin/env python3
"""Inference-only GOLOCO benchmark. Run only in the dedicated measured runtime.

Original .isin selection order, estimator dtype, predict, z-score arithmetic,
and final app.py table construction are preserved. No prediction result is
used as a cache. Reference output exists solely for exact validation.
"""
import argparse
import ast
import contextlib
from collections import Counter, OrderedDict
import gc
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import pickle
import platform
import resource
import sys
import threading
import time
import traceback

import numpy as np
import pandas as pd
import psutil


STAGES = ['input_file_parsing', 'metadata_loading', 'data_preparation',
          'model_loading', 'feature_file_parsing', 'input_selection',
          'estimator_prediction', 'z_score', 'output_construction']


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, obj):
    temp = Path(str(path) + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, default=str) + '\n')
    temp.replace(path)


def append_jsonl(path, obj):
    with open(path, 'a') as handle:
        handle.write(json.dumps(obj, default=str) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


class Deadline(Exception):
    pass


class Timer:
    def __init__(self):
        self.values = {key: 0.0 for key in STAGES}

    @contextlib.contextmanager
    def stage(self, key):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.values[key] = self.values.get(key, 0.0) + time.perf_counter() - start


class Memory:
    """10 ms process-RSS sampler; identical sampler for each compared request."""
    def __enter__(self):
        self.process = psutil.Process()
        self.start = self._rss()
        self.peak = self.start
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def _sample(self):
        while not self.stop.wait(0.01):
            self.peak = max(self.peak, self._rss())

    def _rss(self):
        total = self.process.memory_info().rss
        for child in self.process.children(recursive=True):
            try:
                total += child.memory_info().rss
            except psutil.Error:
                pass
        return total

    def __exit__(self, *args):
        self.end = self._rss()
        self.peak = max(self.peak, self.end)
        self.stop.set()
        self.thread.join()

    def result(self):
        return {'rss_start_bytes': self.start, 'rss_end_bytes': self.end,
                'sampled_peak_rss_bytes': self.peak,
                'rss_change_bytes': self.end - self.start,
                'process_high_water_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1024 if sys.platform != 'darwin' else 1)}


def check_deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise Deadline('Benchmark deadline reached; completed records retained.')


def resolve_model_dir(root):
    root = Path(root)
    if (root / 'L200_models').is_dir():
        return root / 'L200_models'
    if any(root.glob('model_rd10_*.pkl')):
        return root
    found = list(root.rglob('L200_models'))
    if len(found) == 1:
        return found[0]
    raise ValueError('Cannot uniquely locate L200_models in ' + str(root))


def load_author(repo, model_dir, scope, destination, neutral_progress=False):
    spec = importlib.util.spec_from_file_location('goloco_author_session', str(repo / 'app_session.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.gene_stats_dir = str(repo / 'data/19q4_sum_stats.csv')
    module.gene_cats_dir = str(repo / 'data/19q4_gene_cats.csv')
    module.model_dir = str(model_dir.parent)
    # Extract the author's final app inference body without importing Dash or
    # running persistence/dashboard code. Preserve the body through set_index.
    app_source = (repo / 'app.py').read_text()
    tree = ast.parse(app_source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run_inference')
    candidates = [node for node in ast.walk(function) if isinstance(node, ast.If)
                  and any(isinstance(x, ast.Assign) and any(isinstance(y, ast.Name) and y.id == 'scope' for y in x.targets) for x in node.body)]
    body = candidates[0].body
    stop = next(i for i, node in enumerate(body) if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'set_index')
    function.decorator_list = []
    function.body = body[:stop + 1] + [ast.Return(value=ast.Name(id='df_pred', ctx=ast.Load()))]
    excluded_lines = set()
    if neutral_progress:
        def remove_progress(statements):
            kept = []
            for node in statements:
                progress_call = (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                                 and isinstance(node.value.func, ast.Name) and node.value.func.id == 'set_progress')
                progress_assignment = (isinstance(node, ast.Assign) and any(isinstance(x, ast.Name)
                                       and x.id in {'p', 't1', 't2', 't_avg', 'seconds', 'eta'} for x in node.targets))
                if progress_call or progress_assignment:
                    excluded_lines.update(range(node.lineno, node.end_lineno + 1))
                    continue
                if isinstance(node, ast.For):
                    node.body = remove_progress(node.body)
                kept.append(node)
            return kept
        function.body = remove_progress(function.body)
    for node in function.body:
        if isinstance(node, ast.Assign) and any(isinstance(x, ast.Name) and x.id == 'scope' for x in node.targets):
            node.value = ast.Name(id='_goloco_scope', ctx=ast.Load())
    selected = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    if hasattr(ast, 'unparse'):
        extracted = ast.unparse(selected)
    else:
        # Python 3.8 has no ast.unparse. Save executable source with the same
        # single scope substitution used for the compiled AST above.
        all_source_lines = app_source.splitlines()
        source_lines = all_source_lines[body[0].lineno - 1:body[stop].end_lineno]
        indent = body[0].col_offset
        source_lines = [line[indent:] for line in source_lines]
        for node in body[:stop + 1]:
            if isinstance(node, ast.Assign) and any(isinstance(x, ast.Name) and x.id == 'scope' for x in node.targets):
                source_lines[node.lineno - body[0].lineno] = 'scope = _goloco_scope'
        source_lines = [line for i, line in enumerate(source_lines, body[0].lineno) if i not in excluded_lines]
        extracted = ('def run_inference(set_progress, con_n_clicks, L200_filename, L200_data):\n'
                     + '\n'.join('    ' + line for line in source_lines)
                     + '\n    return df_pred')
    (destination / 'author_app_inference_extracted.py').write_text(extracted + '\n')
    namespace = {'pd': pd, 'np': np, 'infer': module.infer, 'logging': logging,
                 'time': time, '_goloco_scope': scope}
    exec(compile(selected, str(repo / 'app.py') + ':inference-only', 'exec'), namespace)
    return module, namespace['run_inference']


def original_output(author_function, frame, deadline=None):
    return author_function(lambda x: check_deadline(deadline), 1, 'example.csv', frame)


def output_table(genes, categories, predictions, z_scores, averages, stds, experiments):
    # Exact app.py construction order and naming, including the ineffective
    # set_index call, are intentionally retained.
    result = pd.DataFrame()
    result['gene'] = genes
    result = pd.merge(result, categories, on='gene', how='left')
    result['gene_category'] = result['gene_category'].replace(np.nan, 'conditional essential')
    result['avg'] = averages
    result['std'] = stds
    for i, exp in enumerate(experiments):
        result[exp + ' (CERES Pred)'] = predictions[:, i]
        result[exp + ' (Z-Score)'] = z_scores[:, i]
    result.set_index('gene')
    return result


def metadata(repo):
    return (pd.read_csv(repo / 'data/19q4_sum_stats.csv', index_col=0),
            pd.read_csv(repo / 'data/19q4_gene_cats.csv'))


def feature_list(model_dir, gene):
    return pd.read_csv(model_dir / ('feats_' + gene + '.csv'), index_col=0)[0:10]['feature'].tolist()


def feature_key(value):
    return ('__goloco_nan__',) if pd.isna(value) else value


class PreparedInput:
    """Per-request preparation; preserves every duplicate input row in row order."""
    def __init__(self, frame, experiments):
        self.matrix = frame[experiments].to_numpy().T
        self.positions = {}
        for i, value in enumerate(frame['feature'].tolist()):
            self.positions.setdefault(feature_key(value), []).append(i)

    def select(self, features):
        found = set()
        for feature in features:
            found.update(self.positions.get(feature_key(feature), []))
        indices = np.array(sorted(found), dtype=np.intp)
        return self.matrix[:, indices], indices


class ModelCache:
    def __init__(self, repo, model_dir, genes, timer=None, deadline=None,
                 max_bytes=16 * 1024 ** 3, process_limit_bytes=None, eager=False):
        timer = timer or Timer()
        self.repo, self.model_dir = Path(repo), Path(model_dir)
        self.deadline = deadline
        self.max_bytes = int(max_bytes)
        self.process_limit_bytes = int(process_limit_bytes or psutil.virtual_memory().available * 0.60)
        with timer.stage('metadata_loading'):
            self.stats, self.categories = metadata(repo)
            # Extract exactly the same NumPy scalars as the original once.
            self.statistics = {gene: (self.stats.loc[gene][0], self.stats.loc[gene][1]) for gene in genes}
        self.models = OrderedDict()
        self.features = {}
        self.sizes = {}
        self.retained_bytes = 0
        self.hits = self.misses = self.evictions = self.invalidations = 0
        self.feature_hits = self.feature_misses = 0
        self.model_type_counts = Counter()
        if eager:
            for gene in genes:
                check_deadline(deadline)
                with timer.stage('model_loading'):
                    self.get_model(gene)
                with timer.stage('feature_file_parsing'):
                    self.get_features(gene)

    def _evict_one(self):
        gene, _ = self.models.popitem(last=False)
        self.retained_bytes -= self.sizes.pop(gene)
        self.evictions += 1

    def get_model(self, gene):
        if gene in self.models:
            self.hits += 1
            self.models.move_to_end(gene)
            return self.models[gene]
        self.misses += 1
        path = self.model_dir / ('model_rd10_' + gene + '.pkl')
        # Conservative accounting of serialized/native tree buffers plus
        # Python overhead. Also enforce the aggregate resident-memory limit.
        charge = max(path.stat().st_size * 2, 65536)
        while self.models and self.retained_bytes + charge > self.max_bytes:
            self._evict_one()
        process = psutil.Process()
        def rss():
            result = process.memory_info().rss
            for child in process.children(recursive=True):
                try:
                    result += child.memory_info().rss
                except psutil.Error:
                    pass
            return result
        if rss() + charge > self.process_limit_bytes:
            while self.models:
                self._evict_one()
            gc.collect()
            if rss() + charge > self.process_limit_bytes:
                raise MemoryError('Conservative aggregate-RSS limit would be exceeded.')
        with open(path, 'rb') as handle:
            model = pickle.load(handle)
        self.model_type_counts[type(model).__module__ + '.' + type(model).__name__] += 1
        if rss() > self.process_limit_bytes:
            raise MemoryError('Aggregate RSS exceeded configured experiment limit.')
        if charge <= self.max_bytes:
            self.models[gene] = model
            self.sizes[gene] = charge
            self.retained_bytes += charge
        return model

    def get_features(self, gene):
        if gene in self.features:
            self.feature_hits += 1
            return self.features[gene]
        self.feature_misses += 1
        result = feature_list(self.model_dir, gene)
        self.features[gene] = result
        return result

    def get_statistics(self, gene):
        if gene not in self.statistics:
            self.statistics[gene] = (self.stats.loc[gene][0], self.stats.loc[gene][1])
        return self.statistics[gene]

    def invalidate(self, gene=None):
        if gene is None:
            self.models.clear()
            self.features.clear()
            self.sizes.clear()
            self.statistics.clear()
            self.retained_bytes = 0
            self.stats, self.categories = metadata(self.repo)
        else:
            self.models.pop(gene, None)
            self.features.pop(gene, None)
            self.statistics.pop(gene, None)
            self.retained_bytes -= self.sizes.pop(gene, 0)
        self.invalidations += 1

    def snapshot(self):
        return {'hits': self.hits, 'misses': self.misses, 'evictions': self.evictions,
                'invalidations': self.invalidations, 'feature_hits': self.feature_hits,
                'feature_misses': self.feature_misses, 'retained_models': len(self.models),
                'retained_accounted_bytes': self.retained_bytes, 'model_cache_budget_bytes': self.max_bytes,
                'aggregate_rss_limit_bytes': self.process_limit_bytes,
                'model_type_load_counts': dict(self.model_type_counts),
                'accounting': 'max(2 * serialized_model_bytes, 65536) per retained model; aggregate RSS including children separately guarded',
                'invalidation_policy': 'explicit invalidate(gene) for model/features/statistic scalar; invalidate() reloads metadata; verified source artifacts immutable'}


def instrumented(frame, genes, repo, model_dir, use_index, cache=None,
                 deadline=None, trace=False):
    timer = Timer()
    traces = {}
    with timer.stage('data_preparation'):
        experiments = frame.columns.tolist()
        experiments.remove('feature')
        prepared = PreparedInput(frame, experiments) if use_index else None
        predictions = np.zeros((len(genes), len(experiments)))
        z_scores = np.zeros_like(predictions)
        averages, stds, output_genes = [], [], []
    with timer.stage('metadata_loading'):
        stats, categories = metadata(repo) if cache is None else (cache.stats, cache.categories)
        prepared_statistics = None
        if cache is None and use_index:
            rows = stats.loc[genes].to_numpy()
            prepared_statistics = {gene: (rows[i, 0], rows[i, 1]) for i, gene in enumerate(genes)}
    for i, gene in enumerate(genes):
        check_deadline(deadline)
        with timer.stage('model_loading'):
            if cache is None:
                with open(model_dir / ('model_rd10_' + gene + '.pkl'), 'rb') as handle:
                    model = pickle.load(handle)
            else:
                model = cache.get_model(gene)
        with timer.stage('feature_file_parsing'):
            features = feature_list(model_dir, gene) if cache is None else cache.get_features(gene)
        with timer.stage('input_selection'):
            if use_index:
                matrix, positions = prepared.select(features)
            else:
                mask = frame.feature.isin(features)
                selection = frame[mask]
                matrix = selection[experiments].to_numpy().T
        with timer.stage('estimator_prediction'):
            pred = model.predict(matrix)
        with timer.stage('z_score'):
            # Preserve the author's positional indexing and float64 arithmetic.
            if cache is None:
                if prepared_statistics is None:
                    average = stats.loc[gene][0]
                    std = stats.loc[gene][1]
                else:
                    average, std = prepared_statistics[gene]
            else:
                average, std = cache.get_statistics(gene)
            z = (pred - average) / std
        with timer.stage('output_construction'):
            output_genes.append(gene)
            predictions[i] = pred
            z_scores[i] = z
            averages.append(average)
            stds.append(std)
        if trace:
            with timer.stage('validation_trace'):
                if not use_index:
                    positions = np.flatnonzero(mask.to_numpy())
                expected_positions = np.flatnonzero(frame.feature.isin(features).to_numpy())
                expected_matrix = frame[frame.feature.isin(features)][experiments].to_numpy().T
                if not np.array_equal(positions, expected_positions):
                    raise AssertionError('Feature selection order mismatch for ' + gene)
                if matrix.dtype != expected_matrix.dtype or not np.array_equal(matrix, expected_matrix, equal_nan=True):
                    raise AssertionError('Estimator input mismatch for ' + gene)
                traces[gene] = {'top_feature_file_rows': features,
                                'input_row_indices': positions.tolist(),
                                'selected_input_features': frame.iloc[positions]['feature'].tolist(),
                                'matrix_dtype': str(matrix.dtype),
                                'matrix_shape': list(matrix.shape),
                                'matrix_sha256': hashlib.sha256(matrix.tobytes(order='C')).hexdigest()}
    with timer.stage('output_construction'):
        result = output_table(output_genes, categories, predictions, z_scores,
                              averages, stds, experiments)
    return result, timer, traces


def predict_optimized(frame, genes, repo, model_dir, use_index=True, cache=None,
                      deadline=None, diagnostics=None):
    """Unprofiled callable backend used by headline benchmark workers.

    All models are called anew. Only ModelCache persists across requests.
    """
    experiments = frame.columns.tolist()
    experiments.remove('feature')
    prepared = PreparedInput(frame, experiments) if use_index else None
    if cache is None:
        stats, categories = metadata(repo)
        # Batched lookup preserves target order and the original row-array
        # coercion. Source metadata has two float64 columns.
        rows = stats.loc[genes].to_numpy()
        statistics = {gene: (rows[i, 0], rows[i, 1]) for i, gene in enumerate(genes)}
    else:
        categories = cache.categories
        statistics = {gene: cache.get_statistics(gene) for gene in genes}
    predictions = np.zeros((len(genes), len(experiments)))
    z_scores = np.zeros_like(predictions)
    averages, stds, output_genes = [], [], []
    for i, gene in enumerate(genes):
        check_deadline(deadline)
        if cache is None:
            with open(Path(model_dir) / ('model_rd10_' + gene + '.pkl'), 'rb') as handle:
                model = pickle.load(handle)
            features = feature_list(model_dir, gene)
        else:
            model = cache.get_model(gene)
            features = cache.get_features(gene)
        if use_index:
            matrix, _ = prepared.select(features)
        else:
            matrix = frame[frame.feature.isin(features)][experiments].to_numpy().T
        pred = model.predict(matrix)
        average, std = statistics[gene]
        z = (pred - average) / std
        output_genes.append(gene)
        predictions[i] = pred
        z_scores[i] = z
        averages.append(average)
        stds.append(std)
        if diagnostics is not None:
            diagnostics['estimator_calls'] = diagnostics.get('estimator_calls', 0) + 1
    return output_table(output_genes, categories, predictions, z_scores,
                        averages, stds, experiments)


class HeadlessGoloco:
    """Minimal callable inference interface with bounded persistent caching."""
    def __init__(self, repo, model_root, max_cache_bytes=16 * 1024 ** 3,
                 process_limit_bytes=None, use_index=True, use_cache=True):
        self.repo = Path(repo)
        self.model_dir = resolve_model_dir(model_root)
        self.max_cache_bytes = max_cache_bytes
        self.process_limit_bytes = process_limit_bytes
        self.use_index, self.use_cache = use_index, use_cache
        self.cache = None

    def predict(self, frame, genes, deadline=None, diagnostics=None):
        if self.use_cache and self.cache is None:
            self.cache = ModelCache(self.repo, self.model_dir, genes,
                                    deadline=deadline, max_bytes=self.max_cache_bytes,
                                    process_limit_bytes=self.process_limit_bytes)
        return predict_optimized(frame, genes, self.repo, self.model_dir,
                                 self.use_index, self.cache, deadline, diagnostics)

    def invalidate(self, gene=None):
        if self.cache is not None:
            self.cache.invalidate(gene)


def validate(reference, candidate, genes, experiments):
    pd.testing.assert_frame_equal(reference, candidate, check_exact=True, check_dtype=True,
                                  check_index_type=True, check_column_type=True)
    if candidate['gene'].tolist() != genes:
        raise AssertionError('Target order differs from requested scope.')
    expected_columns = ['gene', 'gene_category', 'avg', 'std']
    for exp in experiments:
        expected_columns.extend([exp + ' (CERES Pred)', exp + ' (Z-Score)'])
    if candidate.columns.tolist() != expected_columns:
        raise AssertionError('Final app output schema/experiment order mismatch.')
    numeric = candidate.select_dtypes(include=[np.number]).to_numpy()
    reference_numeric = reference.select_dtypes(include=[np.number]).to_numpy()
    finite = np.isfinite(numeric) & np.isfinite(reference_numeric)
    maximum = float(np.max(np.abs(numeric[finite] - reference_numeric[finite]))) if finite.any() else 0.0
    return {'exact_frame_equal': True,
            'target_order_exact': True, 'experiment_order_exact': True,
            'schema_and_dtypes_exact': True, 'max_abs_numeric_difference': maximum,
            'rows': len(candidate), 'columns': list(candidate.columns)}


def describe_model(model):
    result = {'type': type(model).__module__ + '.' + type(model).__name__}
    if hasattr(model, 'get_params'):
        result['params'] = {k: v if isinstance(v, (str, int, float, bool, type(None))) else repr(v)
                            for k, v in model.get_params(deep=False).items()}
    for name in ['n_features_in_', 'n_features_', 'n_outputs_', 'n_estimators']:
        if hasattr(model, name):
            result[name] = getattr(model, name)
    if hasattr(model, 'estimators_'):
        estimators = list(np.ravel(model.estimators_))
        result['estimator_count'] = len(estimators)
        result['estimator_types'] = sorted(set(type(x).__module__ + '.' + type(x).__name__ for x in estimators))
        if estimators and hasattr(estimators[0], 'tree_'):
            result['total_tree_nodes'] = sum(x.tree_.node_count for x in estimators)
    return result


def summary(records, directory):
    requests = [x for x in records if x.get('kind') == 'request']
    if not requests:
        return
    flat = []
    for rec in requests:
        item = {k: v for k, v in rec.items() if k not in ['stages_seconds', 'validation', 'memory']}
        item.update({'stage_' + k + '_seconds': v for k, v in rec.get('stages_seconds', {}).items()})
        item.update(rec['memory'])
        flat.append(item)
    pd.DataFrame(flat).to_csv(directory / 'requests.csv', index=False)
    groups = []
    input_parse = next((x['total_seconds'] for x in records if x.get('kind') == 'input_parse'), 0.0)
    for variant in dict.fromkeys(x['variant'] for x in requests):
        subset = [x for x in requests if x['variant'] == variant]
        repeated = [x['total_seconds'] for x in subset if x['request_index'] > 0]
        first = next((x['total_seconds'] for x in subset if x['request_index'] == 0), None)
        cache_record = next((x for x in records if x.get('kind') == 'cache_build' and x['variant'] == variant), None)
        build = cache_record['total_seconds'] if cache_record else 0.0
        groups.append({'variant': variant, 'targets': subset[0]['targets'],
                       'first_request_seconds': first,
                       'cache_build_seconds': build,
                       'first_request_including_cache_build_seconds': None if first is None else first + build,
                       'input_csv_parse_seconds': input_parse,
                       'first_request_including_cache_build_and_input_csv_parse_seconds': None if first is None else first + build + input_parse,
                       'repeated_requests': len(repeated),
                       'repeated_median_seconds': float(np.median(repeated)) if repeated else None,
                       'repeated_p95_seconds': float(np.percentile(repeated, 95)) if repeated else None,
                       'repeated_std_seconds': float(np.std(repeated)) if repeated else None,
                       'peak_rss_bytes': max(x['memory']['sampled_peak_rss_bytes'] for x in subset + ([cache_record] if cache_record else [])),
                       'all_outputs_exact': all(x['validation']['exact_frame_equal'] for x in subset)})
    pd.DataFrame(groups).to_csv(directory / 'summary.csv', index=False)
    atomic_json(directory / 'summary.json', groups)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--models', '--model-root', dest='models', type=Path, required=True)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--size', default='16', help='16, 1024, 4096, or full; one process per size')
    parser.add_argument('--targets', type=Path, help='Optional ordered JSON list or one-gene-per-line file')
    parser.add_argument('--repeats', type=int, default=3, help='Repeated measured requests after first request')
    parser.add_argument('--variants', default='original,index_only,cache_only,combined')
    parser.add_argument('--deadline-seconds', type=float, default=2400)
    args = parser.parse_args()
    args.repo = args.repo.resolve()
    args.results.mkdir(parents=True, exist_ok=True)
    if (args.results / 'records.jsonl').exists():
        parser.error('Result directory already has records; choose a new result subdirectory.')
    start = time.monotonic()
    deadline = start + args.deadline_seconds
    records = []
    logging.disable(logging.CRITICAL)
    model_dir = resolve_model_dir(args.models)
    all_scope = pd.read_pickle(args.repo / 'data/19q4_genes.pkl')
    canonical = list(all_scope)
    available = set(p.name[len('model_rd10_'):-4] for p in model_dir.glob('model_rd10_*.pkl'))
    features_available = set(p.name[len('feats_'):-4] for p in model_dir.glob('feats_*.csv'))
    supported = [g for g in canonical if g in available and g in features_available]
    if args.targets:
        raw = args.targets.read_text()
        genes = json.loads(raw) if raw.lstrip().startswith('[') else raw.splitlines()
        unknown = [g for g in genes if g not in supported]
        if unknown:
            raise ValueError('Unsupported requested targets: ' + str(unknown[:10]))
    else:
        if args.size != 'full' and int(args.size) > len(supported):
            raise ValueError('Requested %s targets, but only %d supported targets are present.' % (args.size, len(supported)))
        genes = supported if args.size == 'full' else supported[:int(args.size)]
    if not genes:
        raise ValueError('No supported target models were found.')
    atomic_json(args.results / 'targets.json', genes)
    atomic_json(args.results / 'scope.json', {'canonical_count': len(canonical), 'supported_count': len(supported),
                                           'missing_models': [g for g in canonical if g not in available],
                                           'missing_features': [g for g in canonical if g not in features_available]})
    input_path = (args.input or args.repo / 'data/PC9SKMELCOLO_L200_CERES_Features.csv').resolve()
    input_start = time.perf_counter()
    frame = pd.read_csv(input_path)
    input_seconds = time.perf_counter() - input_start
    experiments = [x for x in frame.columns if x != 'feature']
    import sklearn
    import scipy
    import joblib
    environment = {'python': sys.version, 'platform': platform.platform(),
                   'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__,
                   'scipy': scipy.__version__, 'joblib': joblib.__version__,
                   'logical_cpus': os.cpu_count(), 'ram_total_bytes': psutil.virtual_memory().total,
                   'input_file': str(input_path), 'input_sha256': sha256(input_path),
                   'input_parse_seconds': input_seconds, 'input_shape': list(frame.shape),
                   'experiments': experiments, 'app_session_sha256': sha256(args.repo / 'app_session.py'),
                   'app_sha256': sha256(args.repo / 'app.py'), 'args': vars(args),
                   'thread_env': {k: os.environ.get(k) for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']},
                   'timing_policy': 'perf_counter; no OS cache eviction; first means empty application caches; input CSV parsed once from supplied DataFrame API; 10ms process RSS sampling; validation outside request timing',
                   'author_extraction': 'app.run_inference valid-input branch through df_pred.set_index; only scope assignment replaced with requested ordered target list; progress callback is no-op; dashboard/persistence excluded'}
    atomic_json(args.results / 'environment.json', environment)
    module, author_function = load_author(args.repo, model_dir, genes, args.results)
    status = {'state': 'running', 'targets': len(genes)}
    atomic_json(args.results / 'status.json', status)

    def persist(record):
        records.append(record)
        append_jsonl(args.results / 'records.jsonl', record)
        summary(records, args.results)
        print(json.dumps({k: record[k] for k in ['kind', 'variant', 'targets', 'total_seconds'] if k in record}), flush=True)

    try:
        persist({'kind': 'input_parse', 'targets': len(genes), 'total_seconds': input_seconds})
        check_deadline(deadline)
        with Memory() as memory:
            before = time.perf_counter()
            reference = original_output(author_function, frame, deadline)
            elapsed = time.perf_counter() - before
        checks = validate(reference, reference, genes, experiments)
        reference.to_csv(args.results / 'author_original_output.csv', index=False)
        reference.to_pickle(args.results / 'author_original_output.pkl')
        persist({'kind': 'request', 'variant': 'author_original', 'targets': len(genes),
                 'request_index': 0, 'total_seconds': elapsed, 'stages_seconds': {},
                 'memory': memory.result(), 'validation': checks})
        for repeat in range(1, args.repeats + 1):
            check_deadline(deadline)
            with Memory() as memory:
                before = time.perf_counter()
                repeated_reference = original_output(author_function, frame, deadline)
                elapsed = time.perf_counter() - before
            checks = validate(reference, repeated_reference, genes, experiments)
            persist({'kind': 'request', 'variant': 'author_original', 'targets': len(genes),
                     'request_index': repeat, 'total_seconds': elapsed, 'stages_seconds': {},
                     'memory': memory.result(), 'validation': checks})
        if args.repeats:
            del repeated_reference
        # Verify actual serialized model type after the genuine first request.
        with open(model_dir / ('model_rd10_' + genes[0] + '.pkl'), 'rb') as handle:
            first_model = pickle.load(handle)
        atomic_json(args.results / 'model_inspection.json', describe_model(first_model))
        del first_model
        for variant in args.variants.split(','):
            if variant not in ['original', 'index_only', 'cache_only', 'combined']:
                raise ValueError('Unknown variant ' + variant)
            check_deadline(deadline)
            cache = None
            gc.collect()
            if variant in ['cache_only', 'combined']:
                build_timer = Timer()
                with Memory() as memory:
                    before = time.perf_counter()
                    cache = ModelCache(args.repo, model_dir, genes, build_timer, deadline)
                    elapsed = time.perf_counter() - before
                persist({'kind': 'cache_build', 'variant': variant, 'targets': len(genes),
                         'total_seconds': elapsed, 'stages_seconds': build_timer.values,
                         'memory': memory.result(), 'cached_models': len(cache.models),
                         'model_type_counts': dict(Counter(type(m).__module__ + '.' + type(m).__name__ for m in cache.models.values())),
                         'model_attribute_counts': {name: dict(Counter(str(getattr(m, name, 'absent')) for m in cache.models.values()))
                                                    for name in ['n_features_', 'n_features_in_', 'n_estimators', 'n_jobs']},
                         'cache_contains_final_predictions': False})
            # The uninstrumented author path supplies repeated baseline totals;
            # one instrumented baseline request supplies its stage decomposition.
            for repeat in range(1 if variant == 'original' else args.repeats + 1):
                check_deadline(deadline)
                with Memory() as memory:
                    before = time.perf_counter()
                    result, timer, _ = instrumented(frame, genes, args.repo, model_dir,
                                                   variant in ['index_only', 'combined'], cache, deadline)
                    elapsed = time.perf_counter() - before
                checks = validate(reference, result, genes, experiments)
                persist({'kind': 'request', 'variant': variant, 'targets': len(genes),
                         'request_index': repeat, 'total_seconds': elapsed,
                         'stages_seconds': timer.values, 'memory': memory.result(), 'validation': checks})
                if repeat == 0:
                    result.to_csv(args.results / (variant + '_output.csv'), index=False)
            # Validate every selected matrix outside the measured request loop;
            # no estimator calls and no reused predictions participate here.
            check_deadline(deadline)
            prep = PreparedInput(frame, experiments)
            manifest = {}
            for gene in genes:
                check_deadline(deadline)
                features = feature_list(model_dir, gene) if cache is None else cache.features[gene]
                actual, positions = prep.select(features)
                mask = frame.feature.isin(features)
                expected_positions = np.flatnonzero(mask.to_numpy())
                expected = frame[mask][experiments].to_numpy().T
                if not np.array_equal(positions, expected_positions) or actual.dtype != expected.dtype or not np.array_equal(actual, expected, equal_nan=True):
                    raise AssertionError('Exact feature selection validation failed: ' + gene)
                manifest[gene] = {'top_feature_file_rows': features, 'input_row_indices': positions.tolist(),
                                  'selected_input_features': frame.iloc[positions]['feature'].tolist(),
                                  'matrix_dtype': str(actual.dtype), 'matrix_shape': list(actual.shape),
                                  'matrix_sha256': hashlib.sha256(actual.tobytes(order='C')).hexdigest()}
            atomic_json(args.results / (variant + '_selection_manifest.json'), manifest)
            append_jsonl(args.results / 'validation.jsonl', {'variant': variant, 'targets': len(genes),
                                                           'exact_input_matrices': True,
                                                           'all_prediction_tables_exact': True,
                                                           'selection_manifest_sha256': sha256(args.results / (variant + '_selection_manifest.json'))})
            if variant == 'combined' and len(genes) <= 16:
                cases = [('reversed_experiment_columns', frame[['feature'] + list(reversed(experiments))]),
                         ('reversed_input_rows', frame.iloc[::-1].reset_index(drop=True)),
                         ('changed_numeric_input', frame.assign(**{exp: frame[exp] + 0.01 for exp in experiments}))]
                edge_results = []
                for case, altered in cases:
                    check_deadline(deadline)
                    expected = original_output(author_function, altered, deadline)
                    candidate, _, _ = instrumented(altered, genes, args.repo, model_dir, True, cache, deadline)
                    altered_experiments = [x for x in altered.columns if x != 'feature']
                    edge_results.append({'case': case, 'validation': validate(expected, candidate, genes, altered_experiments),
                                         'reuses_model_cache': True, 'recomputes_predictions': True})
                atomic_json(args.results / 'request_freshness_and_order_tests.json', edge_results)
            del cache
            gc.collect()
        status = {'state': 'complete', 'targets': len(genes), 'elapsed_seconds': time.monotonic() - start}
    except Deadline as exc:
        status = {'state': 'deadline', 'targets': len(genes), 'elapsed_seconds': time.monotonic() - start, 'message': str(exc)}
    except Exception as exc:
        status = {'state': 'failed', 'targets': len(genes), 'elapsed_seconds': time.monotonic() - start,
                  'error': repr(exc), 'traceback': traceback.format_exc()}
        raise
    finally:
        summary(records, args.results)
        atomic_json(args.results / 'status.json', status)
        print(json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
