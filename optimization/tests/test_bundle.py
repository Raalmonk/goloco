"""Run only on the authorized Colab VM: python -m unittest discover -s tests.

Tiny numeric fixtures exercise loader integrity without fitting models. Optional
trusted-author conversion tests require GOLOCO_TEST_REPO/MODEL_DIR/MANIFEST.
These tests were authored and syntax checked on the Mac, never executed there.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from goloco_reusable.bundle import (ABI_VERSION, ARRAY_NAMES, BundleError, DTYPES,
                                     FORMAT, FORMAT_VERSION, PIN, PackedBundle,
                                     convert_bundle, verify_bundle)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seal(path, manifest):
    data = (json.dumps(manifest, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    for name in ('manifest.json', 'COMPLETED.json'):
        if (path / name).exists():
            (path / name).chmod(0o644)
    (path / 'manifest.json').write_bytes(data)
    (path / 'COMPLETED.json').write_text(json.dumps({'format_version': FORMAT_VERSION,
        'manifest_sha256': hashlib.sha256(data).hexdigest()}))


def fixture(path):
    path.mkdir()
    # A has two three-node trees; B has one. B's indices remain local, so
    # child1 lives at global offset7, yet its serialized child value is1.
    arrays = {
        'left': [1, -1, -1, 4, -1, -1, 1, -1, -1],
        'right': [2, -1, -1, 5, -1, -1, 2, -1, -1],
        'feature': [0, -2, -2, 0, -2, -2, 0, -2, -2],
        'threshold': [.5, -2., -2., -.5, -2., -2., .25, -2., -2.],
        'value': [0., 1., 2., 0., 3., 4., 0., 5., 6.], 'roots': [0, 3, 0]}
    shards = {}
    for name in ARRAY_NAMES:
        array = np.array(arrays[name], dtype=DTYPES[name])
        file = path / (name + '.npy')
        np.save(file, array, allow_pickle=False)
        shards[name] = {'filename': file.name, 'bytes': file.stat().st_size,
                        'sha256': digest(file), 'shape': list(array.shape), 'dtype': array.dtype.str}
    models = {}
    for gene, no, count, ro, trees in [('A', 0, 6, 0, 2), ('B', 6, 3, 2, 1)]:
        models[gene] = {'node_offset': no, 'root_offset': ro, 'n_nodes': count,
                        'n_trees': trees, 'n_features': 1, 'feature_list': ['f1'],
                        'avg': 0., 'std': 1., 'category': 'conditional essential',
                        'source_model_sha256': 'a' * 64, 'source_model_bytes': 1,
                        'source_feature_sha256': 'b' * 64, 'source_feature_bytes': 1,
                        'model_type': 'sklearn.ensemble._forest.RandomForestRegressor'}
    manifest = {'format': FORMAT, 'format_version': FORMAT_VERSION, 'abi_version': ABI_VERSION,
                'source_commit': PIN, 'endianness': 'little', 'targets': ['A', 'B'],
                'model_order': ['A', 'B'], 'models': models, 'shards': shards,
                'numeric_bytes': sum(len(values) * 8 for values in arrays.values()),
                'metadata_hashes': {'19q4_sum_stats.csv': 'c' * 64, '19q4_gene_cats.csv': 'd' * 64},
                'metadata': {'category_default': 'conditional essential', 'csv_index': True, 'range_index': True}}
    seal(path, manifest)
    return manifest


class BundleRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'bundle'
        self.manifest = fixture(self.path)

    def tearDown(self):
        self.temp.cleanup()

    def mutate_shard(self, name, mutate):
        path = self.path / (name + '.npy')
        array = np.load(path, allow_pickle=False)
        mutate(array)
        np.save(path, array, allow_pickle=False)
        self.manifest['shards'][name].update(bytes=path.stat().st_size, sha256=digest(path))
        seal(self.path, self.manifest)

    def test_load_and_mmap_are_equal_readonly_model_local_views(self):
        with PackedBundle(self.path, mode='load') as ordinary, PackedBundle(self.path, mode='mmap') as mapped:
            self.assertEqual(ordinary.targets, ['A', 'B'])
            self.assertEqual(ordinary.accounted_bytes, mapped.accounted_bytes)
            for gene in ordinary.targets:
                for name in ARRAY_NAMES:
                    a, b = ordinary.get(gene)[name], mapped.arrays_for(gene)[name]
                    np.testing.assert_array_equal(a, b)
                    self.assertFalse(a.flags.writeable)
                    self.assertFalse(b.flags.writeable)
            self.assertEqual(mapped.get('B')['left'].tolist(), [1, -1, -1])
            self.assertEqual(mapped.get('B')['roots'].tolist(), [0])

    def test_close_preserves_issued_owner_views_and_reopen(self):
        bundle = PackedBundle(self.path)
        issued = bundle.get('B')
        bundle.close()
        with self.assertRaises(RuntimeError):
            bundle.get('B')
        self.assertEqual(issued['value'].tolist(), [0., 5., 6.])
        with self.assertRaises(ValueError):
            issued['value'][0] = 100
        with PackedBundle(self.path) as reopened:
            np.testing.assert_array_equal(issued['value'], reopened.get('B')['value'])

    def test_loader_never_unpickles_or_exports(self):
        with mock.patch('pickle.load', side_effect=AssertionError('Bundle loader must not unpickle')):
            with PackedBundle(self.path) as bundle:
                self.assertEqual(bundle.model_info('A')['n_trees'], 2)

    def test_second_fresh_process_loads_without_converter(self):
        code = "from goloco_reusable.bundle import PackedBundle; import pickle; pickle.load=lambda *a,**k:(_ for _ in ()).throw(Exception('unpickle')); b=PackedBundle(__import__('sys').argv[1]); assert b.get('B')['roots'].tolist()==[0]; print('FRESH_PROCESS_BUNDLE_OK'); b.close()"
        process = subprocess.run([sys.executable, '-c', code, str(self.path)], capture_output=True,
                                 text=True, timeout=30, check=True)
        self.assertIn('FRESH_PROCESS_BUNDLE_OK', process.stdout)

    def test_missing_completion_rejects_partial_write(self):
        (self.path / 'COMPLETED.json').unlink()
        with self.assertRaises(BundleError):
            PackedBundle(self.path)

    def test_manifest_checksum_detects_copied_manifest_change(self):
        self.manifest['models']['A']['avg'] = 9.
        (self.path / 'manifest.json').write_text(json.dumps(self.manifest))
        with self.assertRaises(BundleError):
            PackedBundle(self.path)

    def test_wrong_versions_commit_and_endianness(self):
        for field, value in [('format_version', 999), ('abi_version', 999),
                             ('source_commit', '0' * 40), ('endianness', 'big')]:
            with self.subTest(field=field):
                old = self.manifest[field]
                self.manifest[field] = value
                seal(self.path, self.manifest)
                with self.assertRaises(BundleError):
                    PackedBundle(self.path)
                self.manifest[field] = old
        seal(self.path, self.manifest)

    def test_truncated_shard_rejected_before_native_use(self):
        path = self.path / 'value.npy'
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaises(BundleError):
            PackedBundle(self.path)

    def test_corrupt_numeric_byte_rejected_by_checksum(self):
        path = self.path / 'threshold.npy'
        data = bytearray(path.read_bytes())
        data[-1] ^= 1
        path.write_bytes(data)
        with self.assertRaises(BundleError):
            PackedBundle(self.path)

    def test_resealed_cycle_is_structurally_rejected(self):
        self.mutate_shard('left', lambda array: array.__setitem__(0, 0))
        with self.assertRaises(BundleError):
            PackedBundle(self.path)

    def test_resealed_cross_tree_reference_rejected(self):
        self.mutate_shard('right', lambda array: array.__setitem__(0, 4))
        with self.assertRaises(BundleError):
            PackedBundle(self.path)

    def test_resealed_out_of_bounds_feature_rejected(self):
        self.mutate_shard('feature', lambda array: array.__setitem__(0, 1))
        with self.assertRaises(BundleError):
            PackedBundle(self.path)

    def test_resealed_duplicate_parent_rejected(self):
        self.mutate_shard('right', lambda array: array.__setitem__(0, 1))
        with self.assertRaises(BundleError):
            PackedBundle(self.path)

    def test_overlapping_model_offsets_rejected(self):
        self.manifest['models']['B']['node_offset'] = 5
        seal(self.path, self.manifest)
        with self.assertRaises(BundleError):
            PackedBundle(self.path)

    def test_wrong_source_and_metadata_identity_rejected(self):
        expected = {('model_rd10_' + gene + '.pkl'): 'a' * 64 for gene in ['A', 'B']}
        expected.update({('feats_' + gene + '.csv'): 'b' * 64 for gene in ['A', 'B']})
        verify_bundle(self.path, expected_source_manifest=expected)
        expected['model_rd10_A.pkl'] = 'e' * 64
        with self.assertRaises(BundleError):
            PackedBundle(self.path, expected_source_manifest=expected)
        with self.assertRaises(BundleError):
            PackedBundle(self.path, expected_metadata_hashes={'19q4_sum_stats.csv': '0' * 64})

    def test_actual_copied_source_change_rejected(self):
        source = Path(self.temp.name) / 'L200_models'
        source.mkdir()
        for gene in ['A', 'B']:
            for filename, key in [('model_rd10_' + gene + '.pkl', 'source_model_sha256'),
                                  ('feats_' + gene + '.csv', 'source_feature_sha256')]:
                file = source / filename
                file.write_bytes(b'public tiny source identity fixture')
                self.manifest['models'][gene][key] = digest(file)
        seal(self.path, self.manifest)
        verify_bundle(self.path, source_model_root=source)
        (source / 'model_rd10_A.pkl').write_bytes(b'changed copied source')
        with self.assertRaises(BundleError):
            PackedBundle(self.path, source_model_root=source)

    def test_budget_unknown_target_and_repeated_target_order(self):
        with self.assertRaises(BundleError):
            PackedBundle(self.path, max_cache_bytes=1)
        self.manifest['targets'] = ['A', 'B', 'A']
        seal(self.path, self.manifest)
        with PackedBundle(self.path) as bundle:
            self.assertEqual(bundle.targets, ['A', 'B', 'A'])
            with self.assertRaises(KeyError):
                bundle.get('unknown')


@unittest.skipUnless(all(os.environ.get(name) for name in
                        ['GOLOCO_TEST_REPO', 'GOLOCO_TEST_MODEL_DIR', 'GOLOCO_TEST_MANIFEST']),
                     'Trusted author fixture paths not supplied')
class TrustedAuthorConversionTests(unittest.TestCase):
    def test_two_target_conversion_is_deterministic_and_source_verified(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            options = {'repo': os.environ['GOLOCO_TEST_REPO'],
                       'model_dir': os.environ['GOLOCO_TEST_MODEL_DIR'],
                       'source_manifest': os.environ['GOLOCO_TEST_MANIFEST'],
                       'targets': ['A1BG', 'A1CF']}
            one = convert_bundle(out=parent / 'one', **options)
            two = convert_bundle(out=parent / 'two', **options)
            self.assertEqual(one, two)
            for name in ['manifest.json', 'COMPLETED.json'] + [x + '.npy' for x in ARRAY_NAMES]:
                self.assertEqual(digest(parent / 'one' / name), digest(parent / 'two' / name))
            with self.assertRaises(FileExistsError):
                convert_bundle(out=parent / 'one', **options)


if __name__ == '__main__':
    unittest.main()
