#!/usr/bin/env python3
"""Check accepted source hashes and Python syntax; never import inference code."""
import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / 'SCIENTIFIC_SOURCE_MANIFEST.json').read_text())
for entry in manifest['files']:
    path = root / entry['path']
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != entry['published_sha256']:
        raise SystemExit('Source hash mismatch: ' + entry['path'])
    if path.suffix == '.py':
        ast.parse(content, filename=entry['path'])
for path in sorted((root / 'scripts').glob('*.py')):
    ast.parse(path.read_bytes(), filename=str(path.relative_to(root)))
print(json.dumps({'accepted_source_files': len(manifest['files']), 'hashes_match': True,
                  'syntax': 'Python AST only', 'model_execution': False}))
