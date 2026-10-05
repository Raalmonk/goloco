"""Command-line conversion, verification, prediction and original validation."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .backend import HeadlessGoloco, PreparedInput, file_hash, output_table, resolve_model_dir


def targets_from(path, repo):
    if path is None:
        # Only the pinned, trusted author's canonical target pickle is read.
        return list(pd.read_pickle(Path(repo) / 'data/19q4_genes.pkl'))
    value = Path(path).read_text()
    result = json.loads(value) if value.lstrip().startswith('[') else value.splitlines()
    if not isinstance(result, list) or not all(isinstance(x, str) for x in result):
        raise ValueError('Target file must contain a JSON string list or one target per line')
    return result


def original_prediction(repo, model_dir, frame, targets):
    """Fresh author inference methods plus the unchanged final app assembly."""
    repo = Path(repo)
    spec = importlib.util.spec_from_file_location('goloco_cli_author', str(repo / 'app_session.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.gene_stats_dir = str(repo / 'data/19q4_sum_stats.csv')
    module.gene_cats_dir = str(repo / 'data/19q4_gene_cats.csv')
    module.model_dir = str(Path(model_dir).parent)
    author = module.infer(frame)
    predictions = np.zeros((len(targets), len(author.experiments)))
    scores = np.zeros_like(predictions)
    averages, stds = [], []
    for index, gene in enumerate(targets):
        prediction = author.infer_gene(gene)
        average, std, score = author.calc_z_score(gene, prediction)
        predictions[index] = prediction
        scores[index] = score
        averages.append(average)
        stds.append(std)
    return output_table(list(targets), author.gene_categories, predictions, scores,
                        averages, stds, author.experiments)


def validate_backend(backend, frame, targets):
    experiments = frame.columns.tolist()
    experiments.remove('feature')
    prepared = PreparedInput(frame, experiments)
    for gene in targets:
        features = pd.read_csv(backend.model_dir / ('feats_' + gene + '.csv'), index_col=0)[0:10]['feature'].tolist()
        matrix, indices = prepared.select(features)
        mask = frame.feature.isin(features)
        expected = frame[mask][experiments].to_numpy().T
        if (not np.array_equal(indices, np.flatnonzero(mask.to_numpy()))
                or matrix.shape != expected.shape or matrix.dtype != expected.dtype
                or not np.array_equal(matrix, expected, equal_nan=True)):
            raise AssertionError('Selected model matrix differs from original: ' + gene)
    reference = original_prediction(backend.repo, backend.model_dir, frame, targets)
    diagnostics = {}
    candidate = backend.predict(frame, targets, diagnostics=diagnostics)
    exact = True
    exact_error = None
    try:
        pd.testing.assert_frame_equal(reference, candidate, check_exact=True, check_dtype=True,
                                      check_index_type=True, check_column_type=True)
    except AssertionError as exc:
        exact, exact_error = False, str(exc)
    a = reference.select_dtypes(include=[np.number]).to_numpy()
    b = candidate.select_dtypes(include=[np.number]).to_numpy()
    masks_match = (np.array_equal(np.isnan(a), np.isnan(b)) and np.array_equal(np.isposinf(a), np.isposinf(b))
                   and np.array_equal(np.isneginf(a), np.isneginf(b)))
    finite = np.isfinite(a) & np.isfinite(b)
    difference = np.abs(a[finite] - b[finite])
    denom = np.abs(a[finite])
    relative = np.divide(difference, denom, out=np.zeros_like(difference), where=denom != 0)
    relative[(denom == 0) & (difference != 0)] = np.inf
    csv_equal = reference.to_csv() == candidate.to_csv()
    result = {'exact': exact and csv_equal and masks_match, 'exact_frame_equal': exact,
              'exact_error': exact_error, 'serialized_csv_bytes_exact': csv_equal,
              'invalid_masks_match': masks_match, 'atol': 1e-8, 'rtol': 1e-7,
              'within_tolerance': bool(np.allclose(a, b, atol=1e-8, rtol=1e-7, equal_nan=True)),
              'max_absolute_error': float(difference.max()) if len(difference) else 0.0,
              'max_relative_error': float(relative.max()) if len(relative) else 0.0,
              'targets': len(targets), 'experiments': experiments,
              'nan_locations': np.argwhere(np.isnan(b)).tolist(),
              'positive_inf_locations': np.argwhere(np.isposinf(b)).tolist(),
              'negative_inf_locations': np.argwhere(np.isneginf(b)).tolist(),
              'diagnostics': diagnostics, 'cache': backend.snapshot()}
    return result


def main():
    parser = argparse.ArgumentParser(prog='python -m goloco_reusable')
    sub = parser.add_subparsers(dest='command', required=True)
    convert = sub.add_parser('convert', help='Build a verified native bundle from trusted author sources')
    convert.add_argument('--repo', required=True, type=Path)
    convert.add_argument('--models', required=True, type=Path)
    convert.add_argument('--output', required=True, type=Path)
    convert.add_argument('--targets', type=Path)
    convert.add_argument('--source-manifest', type=Path, required=True)
    convert.add_argument('--max-cache-gib', type=float, default=16)
    verify = sub.add_parser('verify', help='Check bundle checksum, format, offsets and forest structure')
    verify.add_argument('--bundle', required=True, type=Path)
    verify.add_argument('--mode', choices=['mmap', 'load'], default='mmap')
    verify.add_argument('--source-manifest', type=Path)
    verify.add_argument('--models', type=Path)
    verify.add_argument('--max-cache-gib', type=float, default=16)
    for name in ['predict', 'validate']:
        command = sub.add_parser(name)
        command.add_argument('--repo', required=True, type=Path)
        command.add_argument('--models', required=True, type=Path)
        command.add_argument('--backend', choices=['numpy', 'python_fast', 'rust'], default='numpy')
        command.add_argument('--bundle', type=Path)
        command.add_argument('--bundle-mode', choices=['mmap', 'load'], default='mmap')
        command.add_argument('--library', type=Path)
        command.add_argument('--input', required=True, type=Path)
        command.add_argument('--targets', type=Path)
        command.add_argument('--max-cache-gib', type=float, default=16)
        command.add_argument('--process-limit-gib', type=float)
        command.add_argument('--chunk-size', type=int, default=1)
        command.add_argument('--invalidate', nargs='?', const='all')
        command.add_argument('--report', type=Path)
        if name == 'predict':
            command.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.command == 'convert':
        from .bundle import convert_bundle
        result = convert_bundle(repo=args.repo, model_dir=resolve_model_dir(args.models),
                                targets=targets_from(args.targets, args.repo), out=args.output,
                                source_manifest=args.source_manifest,
                                max_cache_bytes=int(args.max_cache_gib * 1024 ** 3))
        print(json.dumps({'bundle': str(args.output), 'converted': True,
                          'target_count': len(result['targets']), 'numeric_bytes': result['numeric_bytes'],
                          'manifest': str(args.output / 'manifest.json'),
                          'manifest_sha256': file_hash(args.output / 'manifest.json')}))
        return
    if args.command == 'verify':
        from .bundle import verify_bundle
        result = verify_bundle(args.bundle, mode=args.mode,
                               expected_source_manifest=args.source_manifest,
                               source_model_root=args.models,
                               max_cache_bytes=int(args.max_cache_gib * 1024 ** 3))
        print(json.dumps({'verified': True, 'bundle': str(args.bundle),
                          'target_count': len(result['targets']), 'numeric_bytes': result['numeric_bytes'],
                          'manifest': str(args.bundle / 'manifest.json'),
                          'manifest_sha256': file_hash(args.bundle / 'manifest.json')}))
        return
    targets = targets_from(args.targets, args.repo)
    with HeadlessGoloco(args.repo, args.models, backend=args.backend,
                       model_bundle=args.bundle, bundle_mode=args.bundle_mode, library=args.library,
                       max_cache_bytes=int(args.max_cache_gib * 1024 ** 3),
                       process_limit_bytes=int(args.process_limit_gib * 1024 ** 3) if args.process_limit_gib else None,
                       chunk_size=args.chunk_size) as backend:
        if args.invalidate is not None:
            backend.invalidate(None if args.invalidate == 'all' else args.invalidate)
        if args.command == 'validate':
            result = validate_backend(backend, pd.read_csv(args.input), targets)
        else:
            diagnostics = {}
            frame = backend.predict_csv(args.input, args.output, targets, diagnostics=diagnostics)
            result = {'output': str(args.output), 'rows': len(frame), 'columns': list(frame.columns),
                      'diagnostics': diagnostics, 'cache': backend.snapshot(),
                      'write_completion': 'CSV file closed/flushed; no durable fsync'}
    if args.report:
        args.report.write_text(json.dumps(result, indent=2, default=str) + '\n')
    print(json.dumps(result, default=str))
    if args.command == 'validate' and not result['exact']:
        raise SystemExit(1)
