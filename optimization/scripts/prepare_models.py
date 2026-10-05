#!/usr/bin/env python3
"""Verify official archive bytes and extract only L200 source files, without unpickling.

The destination must not already exist. A failed extraction leaves its partial
folder available for inspection; it does not alter an existing model directory.
This setup helper was source-checked at publication, not rerun on the model archive.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    identity = json.loads((Path(__file__).resolve().parents[1] / 'evidence/model_source_identity.json').read_text())
    if args.output.exists():
        raise SystemExit('Destination already exists; preserve it and choose a new destination.')
    if args.archive.stat().st_size != identity['archive_bytes']:
        raise SystemExit('Official archive byte count mismatch')
    sha = hashlib.sha256()
    md5 = hashlib.md5()
    with args.archive.open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b''):
            sha.update(chunk)
            md5.update(chunk)
    if sha.hexdigest() != identity['archive_sha256'] or md5.hexdigest() != identity['archive_md5']:
        raise SystemExit('Official archive checksum mismatch')
    args.output.mkdir(parents=True)
    model_dir = args.output / 'L200_models'
    model_dir.mkdir()
    files = []
    seen = set()
    with zipfile.ZipFile(args.archive) as archive:
        for info in archive.infolist():
            parts = PurePosixPath(info.filename).parts
            if ('L200_models' not in parts or info.is_dir() or '__MACOSX' in parts):
                continue
            if (PurePosixPath(info.filename).is_absolute() or '..' in parts
                    or '\\' in info.filename or stat.S_ISLNK(info.external_attr >> 16)):
                raise ValueError('Unsafe archive member')
            name = parts[-1]
            if not (name.startswith('model_rd10_') and name.endswith('.pkl')
                    or name.startswith('feats_') and name.endswith('.csv')):
                continue
            if name in seen:
                raise ValueError('Duplicate source filename')
            seen.add(name)
            dest = model_dir / name
            checksum = hashlib.sha256()
            count = 0
            with archive.open(info) as source, dest.open('xb') as target:
                for chunk in iter(lambda: source.read(8 * 1024**2), b''):
                    target.write(chunk)
                    checksum.update(chunk)
                    count += len(chunk)
            if count != info.file_size:
                raise ValueError('Extracted source length mismatch')
            dest.chmod(0o444)
            files.append({'path': name, 'bytes': count, 'sha256': checksum.hexdigest()})
    files.sort(key=lambda x: x['path'])
    inventory_bytes = json.dumps(files, sort_keys=True, separators=(',', ':')).encode()
    if (len(files) != identity['source_file_count'] or
            hashlib.sha256(inventory_bytes).hexdigest() != identity['source_inventory_sha256']):
        raise SystemExit('Extracted source inventory differs from accepted author models')
    manifest = args.output / 'author_model_manifest.json'
    manifest.write_text(json.dumps({'files': files}, indent=2) + '\n')
    print(json.dumps({'verified_archive': True, 'files': len(files),
                      'manifest': str(manifest), 'model_directory': str(model_dir)}))


if __name__ == '__main__':
    main()
