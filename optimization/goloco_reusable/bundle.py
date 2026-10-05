"""Verified numeric GOLOCO bundles; no model deserialization during loading.

Six global NPY arrays contain contiguous per-model slices. Child/root indices
are LOCAL to each model slice, exactly as in the accepted C0 adapter. Opening
always hashes every shard and validates every forest; these are startup costs.
Issued read-only views retain their NumPy/mmap owners after loader.close().
"""
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
import uuid

import numpy as np

PIN = '470477c4be6ba0095016a2089b9bbf33d49a1ef8'
FORMAT = 'goloco-native-arrays'
FORMAT_VERSION = 1
ABI_VERSION = 1
DEFAULT_BUDGET = 16 * 1024 ** 3
ARRAY_NAMES = ('left', 'right', 'feature', 'threshold', 'value', 'roots')
DTYPES = {name: '<i8' if name in ('left', 'right', 'feature', 'roots') else '<f8'
          for name in ARRAY_NAMES}


class BundleError(ValueError):
    """Incompatible, incomplete, structurally invalid, or changed bundle."""


def _require(condition, message):
    if not condition:
        raise BundleError(message)


def _digest(path):
    h = hashlib.sha256()
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(), 'Missing regular file: ' + str(path))
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False,
                       ensure_ascii=False) + '\n').encode('utf-8')


def _json_file(path):
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(), 'Missing regular file: ' + str(path))
    _require(path.stat().st_size <= 128 * 1024 ** 2, 'Oversized manifest: ' + str(path))
    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise BundleError('Duplicate JSON key: ' + key)
            result[key] = value
        return result
    try:
        return json.loads(path.read_text(), object_pairs_hook=no_duplicates,
                          parse_constant=lambda value: (_ for _ in ()).throw(BundleError('Nonfinite JSON value')))
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        raise BundleError('Invalid JSON file: ' + str(path)) from exc


def _integer(value, label, minimum=0):
    _require(isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
             'Invalid integer ' + label)
    _require(value <= np.iinfo(np.int64).max, 'Integer overflow in ' + label)
    return value


def _target(gene):
    _require(isinstance(gene, str) and gene and Path(gene).name == gene
             and gene not in ('.', '..') and '\\' not in gene and '\x00' not in gene,
             'Invalid target ID')
    return gene


def _source_entries(source_manifest):
    if source_manifest is None:
        return {}
    document = _json_file(source_manifest) if isinstance(source_manifest, (str, Path)) else source_manifest
    _require(isinstance(document, dict), 'Source manifest must be an object')
    entries = document.get('files')
    result = {}
    if entries is None:
        entries = [{'path': name, **({'sha256': value} if isinstance(value, str) else value)}
                   for name, value in document.items()]
    _require(isinstance(entries, list), 'Source manifest files must be a list')
    for entry in entries:
        _require(isinstance(entry, dict) and isinstance(entry.get('path'), str), 'Malformed source entry')
        name = Path(entry['path']).name
        checksum = entry.get('sha256')
        _require(isinstance(checksum, str) and len(checksum) == 64
                 and all(c in '0123456789abcdef' for c in checksum), 'Malformed source checksum')
        normalized = {'sha256': checksum}
        if 'bytes' in entry:
            normalized['bytes'] = _integer(entry['bytes'], 'source bytes')
        if name in result:
            _require(result[name] == normalized, 'Ambiguous source filename: ' + name)
        result[name] = normalized
    return result


def _hashes_by_basename(values):
    return {Path(name).name: value['sha256'] if isinstance(value, dict) else value
            for name, value in values.items()}


def _model_dir(root):
    root = Path(root)
    if (root / 'L200_models').is_dir():
        return root / 'L200_models'
    if any(root.glob('model_rd10_*.pkl')):
        return root
    found = list(root.rglob('L200_models'))
    if len(found) == 1:
        return found[0]
    raise BundleError('Cannot uniquely locate L200_models')


def _verified_source(path, expected):
    _require(path.is_file() and not path.is_symlink(), 'Missing regular source: ' + str(path))
    before = path.stat()
    if expected and 'bytes' in expected:
        _require(before.st_size == expected['bytes'], 'Source length changed: ' + path.name)
    checksum = _digest(path)
    if expected:
        _require(checksum == expected['sha256'], 'Source checksum changed: ' + path.name)
    after = path.stat()
    _require((before.st_size, before.st_mtime_ns, before.st_ino) ==
             (after.st_size, after.st_mtime_ns, after.st_ino), 'Source changed during verification')
    return {'sha256': checksum, 'bytes': before.st_size}


def validate_arrays(arrays, info):
    """Vectorized proof of valid, connected, acyclic per-tree traversal graphs.

    The format retains sklearn's original forward node numbering. Forward
    children imply acyclicity; unique parents and isolated root ranges prove
    each node belongs to exactly one connected tree. No recursive Python walk.
    """
    n = _integer(info['n_nodes'], 'n_nodes', 1)
    trees = _integer(info['n_trees'], 'n_trees', 1)
    width = _integer(info['n_features'], 'n_features', 1)
    for name in ARRAY_NAMES:
        array = arrays[name]
        length = trees if name == 'roots' else n
        _require(array.ndim == 1 and array.shape == (length,), 'Array shape mismatch: ' + name)
        _require(array.dtype.str == DTYPES[name] and array.dtype.isnative,
                 'Array dtype/endianness mismatch: ' + name)
        _require(array.flags.c_contiguous and array.flags.aligned, 'Unaligned/noncontiguous array: ' + name)
    roots, left, right, feature = (arrays[x] for x in ('roots', 'left', 'right', 'feature'))
    _require(roots[0] == 0 and np.all(roots >= 0) and np.all(roots < n)
             and np.all(roots[1:] > roots[:-1]), 'Invalid ordered roots')
    leaves = left == -1
    _require(np.array_equal(leaves, right == -1), 'Mismatched leaf children')
    internal = ~leaves
    positions = np.flatnonzero(internal)
    tree_ends = np.r_[roots[1:], np.int64(n)]
    ends = tree_ends[np.searchsorted(roots, positions, side='right') - 1]
    for child in (left[internal], right[internal]):
        _require(np.all(child > positions) and np.all(child < ends),
                 'Cyclic, backward, or cross-tree child index')
    _require(np.all(feature[internal] >= 0) and np.all(feature[internal] < width),
             'Internal feature index out of bounds')
    children = np.concatenate((left[internal], right[internal]))
    parents = np.bincount(children, minlength=n)
    expected = np.ones(n, dtype=parents.dtype)
    expected[roots] = 0
    _require(np.array_equal(parents, expected), 'Duplicate parent or unreachable node')
    return True


class PackedBundle:
    """Single-caller verified loader. Views safely outlive close().

    close() rejects future lookups and releases this loader's references.
    Returned NumPy views keep their base owners alive; mappings are released
    when the last view/adapter is released. Never force-close an in-flight map.
    """
    def __init__(self, path, mode='mmap', expected_commit=PIN,
                 expected_source_manifest=None, max_cache_bytes=DEFAULT_BUDGET,
                 expected_source_commit=None, expected_metadata_hashes=None,
                 source_model_root=None):
        self.path, self.mode = Path(path), mode
        self.closed, self._arrays = False, {}
        _require(mode in ('mmap', 'load'), "mode must be 'mmap' or 'load'")
        _require(sys.byteorder == 'little', 'Bundle ABI requires little-endian host')
        try:
            completed = _json_file(self.path / 'COMPLETED.json')
            manifest_path = self.path / 'manifest.json'
            _require(completed.get('format_version') == FORMAT_VERSION, 'Unsupported completion format version')
            _require(_digest(manifest_path) == completed.get('manifest_sha256'), 'Manifest checksum mismatch/partial write')
            manifest = _json_file(manifest_path)
            _require(manifest.get('format') == FORMAT and manifest.get('format_version') == FORMAT_VERSION,
                     'Unsupported bundle format/version')
            _require(manifest.get('abi_version') == ABI_VERSION and manifest.get('endianness') == 'little',
                     'Incompatible native ABI/endianness')
            expected_commit = expected_source_commit or expected_commit
            if expected_commit is not None:
                _require(manifest.get('source_commit') == expected_commit, 'Source commit mismatch')
            self.manifest = manifest
            self.targets = list(manifest['targets'])
            for gene in self.targets:
                _target(gene)
            order = list(dict.fromkeys(self.targets))
            _require(order and manifest.get('model_order') == order and set(manifest['models']) == set(order),
                     'Target/model order or membership mismatch')
            self.metadata = dict(manifest['metadata'])
            self.metadata['metadata_hashes'] = dict(manifest['metadata_hashes'])
            if expected_metadata_hashes is not None:
                _require(_hashes_by_basename(expected_metadata_hashes) == manifest['metadata_hashes'],
                         'Reference metadata identity mismatch')
            expected_sources = _source_entries(expected_source_manifest)
            if expected_sources:
                for gene in order:
                    info = manifest['models'][gene]
                    for name, key in [('model_rd10_' + gene + '.pkl', 'source_model_sha256'),
                                      ('feats_' + gene + '.csv', 'source_feature_sha256')]:
                        _require(name in expected_sources and expected_sources[name]['sha256'] == info[key],
                                 'Expected source manifest differs for ' + name)
            _require(set(manifest['shards']) == set(ARRAY_NAMES), 'Missing/unexpected array shards')
            numeric_bytes = 0
            for name in ARRAY_NAMES:
                shard = manifest['shards'][name]
                _require(shard['filename'] == name + '.npy', 'Unsafe/unexpected shard filename')
                _require(shard['dtype'] == DTYPES[name] and len(shard['shape']) == 1, 'Invalid shard dtype/shape')
                count = _integer(shard['shape'][0], name + ' length', 1)
                numeric_bytes += count * 8
            self.mapped_or_loaded_bytes = numeric_bytes
            self.accounted_bytes = numeric_bytes + manifest_path.stat().st_size * 4
            _require(self.accounted_bytes <= int(max_cache_bytes), 'Bundle exceeds declared cache budget')
            for name in ARRAY_NAMES:
                shard = manifest['shards'][name]
                file = self.path / shard['filename']
                _require(file.is_file() and not file.is_symlink(), 'Missing regular shard: ' + name)
                _require(file.stat().st_size == _integer(shard['bytes'], name + ' file bytes', 1),
                         'Truncated/extended shard: ' + name)
                _require(_digest(file) == shard['sha256'], 'Shard checksum mismatch: ' + name)
                try:
                    array = np.load(file, mmap_mode='r' if mode == 'mmap' else None, allow_pickle=False)
                except (OSError, ValueError, EOFError) as exc:
                    raise BundleError('Invalid numeric NPY shard: ' + name) from exc
                _require(array.shape == tuple(shard['shape']) and array.dtype.str == shard['dtype']
                         and array.dtype.isnative and array.ndim == 1, 'NPY header disagrees with manifest')
                array.flags.writeable = False
                self._arrays[name] = array
            node_offset = root_offset = 0
            for gene in order:
                info = manifest['models'][gene]
                _require(info['node_offset'] == node_offset and info['root_offset'] == root_offset,
                         'Overlapping, gapped, or unordered model offsets')
                n, t = _integer(info['n_nodes'], 'model nodes', 1), _integer(info['n_trees'], 'model trees', 1)
                node_offset += n
                root_offset += t
                _require(node_offset <= self._arrays['left'].size and root_offset <= self._arrays['roots'].size,
                         'Model slice outside shards')
                _require(isinstance(info['feature_list'], list) and len(info['feature_list']) <= 10,
                         'Invalid top-ten feature metadata')
                validate_arrays(self._views(gene), info)
            _require(all(self._arrays[name].size == node_offset for name in ARRAY_NAMES if name != 'roots')
                     and self._arrays['roots'].size == root_offset, 'Trailing/mismatched shard elements')
            if source_model_root is not None:
                directory = _model_dir(source_model_root)
                for gene in order:
                    info = manifest['models'][gene]
                    for filename, key in [('model_rd10_' + gene + '.pkl', 'source_model_sha256'),
                                          ('feats_' + gene + '.csv', 'source_feature_sha256')]:
                        _verified_source(directory / filename, {'sha256': info[key]})
        except BaseException:
            self.close()
            raise

    def _views(self, gene):
        info = self.manifest['models'][gene]
        node = slice(info['node_offset'], info['node_offset'] + info['n_nodes'])
        roots = slice(info['root_offset'], info['root_offset'] + info['n_trees'])
        return {name: array[roots if name == 'roots' else node] for name, array in self._arrays.items()}

    def get(self, gene):
        if self.closed:
            raise RuntimeError('Bundle is closed; open a new loader before requesting arrays')
        if gene not in self.manifest['models']:
            raise KeyError(gene)
        return self._views(gene)

    arrays_for = get

    def model_info(self, gene):
        if self.closed:
            raise RuntimeError('Bundle is closed')
        return self.manifest['models'][gene]

    def close(self):
        self.closed = True
        self._arrays.clear()

    def __enter__(self):
        if self.closed:
            raise RuntimeError('Bundle is closed')
        return self

    def __exit__(self, *args):
        self.close()


NativeBundle = PackedBundle


def verify_bundle(path, **kwargs):
    with PackedBundle(path, **kwargs) as bundle:
        return bundle.manifest


def _guard_model(model):
    # Imported only during trusted-source conversion, never bundle loading.
    import joblib
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.utils.validation import check_is_fitted
    _require(type(model) is RandomForestRegressor, 'Unsupported source model class')
    check_is_fitted(model)
    _require(model.n_outputs_ == 1 and joblib.effective_n_jobs(model.n_jobs) == 1,
             'Source model is outside the accepted single-output/single-worker scope')
    _require(model.estimators_, 'Empty source forest')
    for tree in model.estimators_:
        _require(type(tree) is DecisionTreeRegressor and tree.n_outputs_ == 1
                 and tree.n_features_ == model.n_features_
                 and tree.tree_.n_features == model.n_features_ and tree.tree_.n_outputs == 1
                 and tree.tree_.value.shape[1:] == (1, 1),
                 'Incompatible source tree shape/type')
    return sum(tree.tree_.node_count for tree in model.estimators_), len(model.estimators_)


def _rss_guard(limit):
    import psutil
    root = psutil.Process()
    resident = root.memory_info().rss
    for child in root.children(recursive=True):
        try:
            resident += child.memory_info().rss
        except psutil.Error:
            pass
    if resident > limit:
        raise MemoryError('Bundle conversion exceeded aggregate process RSS limit')
    return resident


def convert_bundle(repo, model_dir=None, targets=None, out=None, source_manifest=None,
                   max_cache_bytes=DEFAULT_BUDGET, model_root=None, output=None,
                   expected_commit=PIN, max_rss_bytes=None):
    """Two-pass bounded conversion from verified trusted source models.

    Only one original model is live at a time. Six file-backed arrays are
    published by one directory rename after complete checksum/structure checks.
    Existing completed bundles are preserved. All manifest bytes are deterministic.
    """
    import pandas as pd
    import psutil
    repo = Path(repo)
    directory = _model_dir(model_dir if model_dir is not None else model_root)
    destination = Path(out if out is not None else output)
    if destination.exists():
        raise FileExistsError('Bundle destination already exists: ' + str(destination))
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    _require(commit == expected_commit, 'Converter source commit mismatch')
    _require(not subprocess.check_output(['git', '-C', str(repo), 'diff', '--name-only'], text=True).strip()
             and not subprocess.check_output(['git', '-C', str(repo), 'diff', '--cached', '--name-only'], text=True).strip(),
             'Converter requires unchanged tracked source')
    if targets is None:
        targets = list(pd.read_pickle(repo / 'data/19q4_genes.pkl'))
    targets = [_target(gene) for gene in targets]
    order = list(dict.fromkeys(targets))
    _require(order, 'Cannot build an empty bundle')
    sources = _source_entries(source_manifest)
    metadata_files = [repo / 'data/19q4_sum_stats.csv', repo / 'data/19q4_gene_cats.csv']
    metadata_hashes = {path.name: _digest(path) for path in metadata_files}
    stats = pd.read_csv(metadata_files[0], index_col=0)
    cats = pd.read_csv(metadata_files[1])
    _require(stats.index.is_unique and cats['gene'].is_unique, 'Ambiguous reference metadata')
    categories = dict(zip(cats['gene'], cats['gene_category']))
    process_limit = min(int(max_rss_bytes or psutil.virtual_memory().available * .60),
                        int(psutil.virtual_memory().available * .60))
    started = time.perf_counter()
    models, node_offset, root_offset = {}, 0, 0
    for index, gene in enumerate(order):
        model_path, feature_path = directory / ('model_rd10_' + gene + '.pkl'), directory / ('feats_' + gene + '.csv')
        expected_model, expected_feature = sources.get(model_path.name), sources.get(feature_path.name)
        if sources:
            _require(expected_model is not None and expected_feature is not None, 'Source manifest lacks target ' + gene)
        model_source = _verified_source(model_path, expected_model)
        feature_source = _verified_source(feature_path, expected_feature)
        features = pd.read_csv(feature_path, index_col=0).iloc[:10]['feature'].tolist()
        _require(all(isinstance(value, str) for value in features), 'Bundle feature IDs must be strings')
        row = stats.loc[gene]
        average, standard_deviation = float(row.iloc[0]), float(row.iloc[1])
        category = categories.get(gene, 'conditional essential')
        if pd.isna(category):
            category = 'conditional essential'
        with model_path.open('rb') as stream:
            model = pickle.load(stream)
        nodes, trees = _guard_model(model)
        models[gene] = {'node_offset': node_offset, 'root_offset': root_offset,
                        'n_nodes': int(nodes), 'n_trees': int(trees), 'n_features': int(model.n_features_),
                        'feature_list': features, 'avg': average, 'std': standard_deviation, 'category': category,
                        'source_model_sha256': model_source['sha256'], 'source_model_bytes': model_source['bytes'],
                        'source_feature_sha256': feature_source['sha256'], 'source_feature_bytes': feature_source['bytes'],
                        'model_type': type(model).__module__ + '.' + type(model).__name__}
        node_offset += nodes
        root_offset += trees
        del model
        _rss_guard(process_limit)
        if index % 1000 == 0:
            print('BUNDLE_SOURCE_SCAN', index, len(order), flush=True)
    numeric_bytes = node_offset * 5 * 8 + root_offset * 8
    _require(numeric_bytes <= int(max_cache_bytes), 'Numeric bundle exceeds cache budget')
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / ('.' + destination.name + '.partial-' + uuid.uuid4().hex)
    staging.mkdir()
    arrays = {}
    try:
        for name in ARRAY_NAMES:
            length = root_offset if name == 'roots' else node_offset
            arrays[name] = np.lib.format.open_memmap(staging / (name + '.npy'), mode='w+', dtype=DTYPES[name], shape=(length,))
        for index, gene in enumerate(order):
            info = models[gene]
            path = directory / ('model_rd10_' + gene + '.pkl')
            _verified_source(path, {'sha256': info['source_model_sha256'], 'bytes': info['source_model_bytes']})
            _verified_source(directory / ('feats_' + gene + '.csv'),
                             {'sha256': info['source_feature_sha256'], 'bytes': info['source_feature_bytes']})
            with path.open('rb') as stream:
                model = pickle.load(stream)
            _require(_guard_model(model) == (info['n_nodes'], info['n_trees']), 'Source shape changed between passes')
            offset = 0
            for tree_index, tree in enumerate(model.estimators_):
                native, count = tree.tree_, tree.tree_.node_count
                section = slice(info['node_offset'] + offset, info['node_offset'] + offset + count)
                arrays['roots'][info['root_offset'] + tree_index] = offset
                arrays['left'][section] = np.where(native.children_left == -1, -1, native.children_left + offset)
                arrays['right'][section] = np.where(native.children_right == -1, -1, native.children_right + offset)
                arrays['feature'][section] = native.feature
                arrays['threshold'][section] = native.threshold
                arrays['value'][section] = native.value[:, 0, 0]
                offset += count
            views = {name: array[info['root_offset']:info['root_offset'] + info['n_trees']]
                     if name == 'roots' else array[info['node_offset']:info['node_offset'] + info['n_nodes']]
                     for name, array in arrays.items()}
            validate_arrays(views, info)
            del model, views, tree, native
            _rss_guard(process_limit)
            if index % 1000 == 0:
                print('BUNDLE_EXPORT', index, len(order), flush=True)
        for array in arrays.values():
            array.flush()
        del array
        arrays.clear()
        gc.collect()
        shards = {}
        for name in ARRAY_NAMES:
            path = staging / (name + '.npy')
            with path.open('rb') as stream:
                os.fsync(stream.fileno())
            path.chmod(0o444)
            shards[name] = {'filename': path.name, 'bytes': path.stat().st_size, 'sha256': _digest(path),
                            'dtype': DTYPES[name], 'shape': [root_offset if name == 'roots' else node_offset]}
        for path in metadata_files:
            _require(_digest(path) == metadata_hashes[path.name], 'Reference metadata changed during conversion')
        manifest = {'format': FORMAT, 'format_version': FORMAT_VERSION, 'abi_version': ABI_VERSION,
                    'source_commit': commit, 'endianness': 'little', 'targets': targets, 'model_order': order,
                    'models': models, 'shards': shards, 'numeric_bytes': numeric_bytes,
                    'metadata_hashes': metadata_hashes,
                    'metadata': {'category_default': 'conditional essential', 'csv_index': True, 'range_index': True,
                                 'statistics_columns': stats.columns.tolist()},
                    'validation_policy': 'Full SHA-256 on each open; strict numeric NPY; ordered contiguous model slices; forward acyclic rooted trees; no unpickling on open',
                    'source_validator_dependency': 'Original unchanged sklearn estimators remain required for legacy input validation'}
        manifest_data = _json_bytes(manifest)
        _require(numeric_bytes + len(manifest_data) * 4 <= int(max_cache_bytes), 'Bundle plus manifest exceeds cache budget')
        (staging / 'manifest.json').write_bytes(manifest_data)
        (staging / 'COMPLETED.json').write_bytes(_json_bytes({'format_version': FORMAT_VERSION,
            'manifest_sha256': hashlib.sha256(manifest_data).hexdigest()}))
        for name in ('manifest.json', 'COMPLETED.json'):
            with (staging / name).open('rb') as stream:
                os.fsync(stream.fileno())
            (staging / name).chmod(0o444)
        # Loader exercises the exact published format without re-exporting.
        verify_bundle(staging, expected_commit=commit, expected_source_manifest=source_manifest,
                      expected_metadata_hashes=metadata_hashes, max_cache_bytes=max_cache_bytes)
        _require(not destination.exists(), 'Destination appeared during conversion')
        staging.rename(destination)
        print('BUNDLE_COMPLETE', json.dumps({'targets': len(order), 'numeric_bytes': numeric_bytes,
              'seconds': time.perf_counter() - started, 'manifest_sha256': hashlib.sha256(manifest_data).hexdigest()}), flush=True)
        return manifest
    except BaseException:
        arrays.clear()
        # Failed staging remains explicitly unpublished and diagnosable.
        raise
