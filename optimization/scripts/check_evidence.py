#!/usr/bin/env python3
"""Lightweight publication checks; no inference imports or model execution."""
import ast
import hashlib
import importlib.util
import json
import re
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('evidence_summary',ROOT/'benchmarks/summarize.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
records=m.read_records();summary=m.summarize(records)
assert summary==json.loads((ROOT/'figures/source_tables/summary.json').read_text())
manifest=json.loads((ROOT/'benchmarks/data/export_manifest.json').read_text())
hex64=re.compile(r'^[0-9a-f]{64}$')
assert hex64.fullmatch(manifest['source_sha256'])
assert all(hex64.fullmatch(v['sha256']) for v in manifest['source_files'].values())
assert sum(v['records_in_export'] for v in manifest['source_files'].values())==92
for x in records:
    assert hex64.fullmatch(x['source_line_sha256'])
    assert x['source'] in manifest['source_files']
    for cache in ('cache_before','cache_after'):
        model_bundle=x['record'].get(cache,{}).get('model_bundle')
        assert model_bundle in (None,'artifact:native_bundle')
for filename, info in manifest['supplemental_derivatives'].items():
    assert m.digest(ROOT/'benchmarks/data'/filename)==info['sha256']
context=json.loads((ROOT/'benchmarks/data/context.json').read_text())
for value in context['records'].values():
    assert hex64.fullmatch(value['source_sha256'])
    assert sorted(value['data'])==sorted(value['retained_fields'])
assert context['records']['bundle_conversion']['data']['returncode']==0
assert context['records']['matrix_audit']['data']['all_matrices_exact'] is True
assert context['records']['backend_contracts']['data']['failed']==[]
assert context['records']['chunk_contracts']['data']['all_exact_or_same_exception_type'] is True
public_text=(ROOT/'benchmarks/data/phase2_timings.jsonl').read_text()+(ROOT/'benchmarks/data/context.json').read_text()
assert not re.search(r'(?:/Users/|/content/|/private/|https?://|session[_-]id|process_group_id|deadline_epoch)',public_text)
provenance=json.loads((ROOT/'figures/provenance.json').read_text())
expected={'matched_residency':10,'reusable_resident':20,'startup_latency':12,'standalone_memory':20,'negative_chunking':15}
assert {x['figure'] for x in provenance['figures']}==set(expected)
ids={x['id']:x for x in records}
figure_bytes=0
for entry in provenance['figures']:
    stem=entry['figure'];assert len(entry['record_ids'])==expected[stem]
    assert entry['data_sha256']==m.digest(ROOT/'benchmarks/data/phase2_timings.jsonl')
    rows=[ids[x] for x in entry['record_ids']]
    assert all(x['target_count']==18141 and x['experiment_count']==3 for x in rows)
    assert set(entry['source_files'])==set(x['source'] for x in rows)
    for source,digest in entry['original_source_sha256'].items():
        assert digest==manifest['source_files'][source]['sha256']
    for ext in ('svg','pdf','png'):
        p=ROOT/'figures'/(stem+'.'+ext);assert p.is_file();assert p.stat().st_size<10*1024**2
        figure_bytes+=p.stat().st_size
        data=p.read_bytes()
        if ext=='svg':
            tree=ET.fromstring(data);assert tree.findall('.//{http://www.w3.org/2000/svg}text')
            assert not tree.findall('.//{http://www.w3.org/2000/svg}image'), 'SVG should contain vector marks, not embedded raster chart'
        elif ext=='pdf':
            assert data.startswith(b'%PDF-')
            assert b'/Subtype /Image' not in data, 'PDF should contain vector chart marks'
        else:
            assert data.startswith(b'\x89PNG\r\n\x1a\n')
            width,height=struct.unpack('>II',data[16:24]);assert width>=2400 and height>=1000
            index=data.find(b'pHYs');assert index>0
            ppm_x,ppm_y,unit=struct.unpack('>IIB',data[index+4:index+13])
            assert unit==1 and abs(ppm_x*.0254-300)<.1 and ppm_x==ppm_y
# Syntax-only checks do not import any inference or legacy dependencies.
python_files=list(ROOT.rglob('*.py'))
for path in python_files:ast.parse(path.read_text(),filename=str(path.relative_to(ROOT)))
print(json.dumps({'publication_checks':'passed','timed_records':92,'correctness':summary['correctness'],
                  'figures':5,'figure_files':15,'figure_bytes':figure_bytes,'python_files_parsed':len(python_files),
                  'model_dependent_tests_executed':False},sort_keys=True))
