#!/usr/bin/env python3
"""Summarize committed evidence using only the Python standard library.

No inference modules, models, numeric bundles, or network access are used.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'benchmarks' / 'data'
PACKAGED = ('source_python_fast', 'source_rust', 'packed_load', 'packed_mmap')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_records(data_dir=DATA):
    return [json.loads(line) for line in (data_dir / 'phase2_timings.jsonl').read_text().splitlines()]

def quartile(values, q):
    """Linear interpolation at (n - 1) * q; matches NumPy method='linear'."""
    values = sorted(values)
    position = (len(values) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)

def stats(values):
    return {'n': len(values), 'median': median(values), 'min': min(values), 'max': max(values),
            'q1': quartile(values, .25), 'q3': quartile(values, .75),
            'iqr': quartile(values, .75) - quartile(values, .25)}

def validate(records, data_dir=DATA):
    manifest = json.loads((data_dir / 'export_manifest.json').read_text())
    assert digest(data_dir / manifest['public_file']) == manifest['public_sha256']
    assert len(records) == manifest['record_count'] == 92
    assert [x['id'] for x in records] == ['p2-%03d' % i for i in range(1, 93)]
    assert len({(x['source'], x['record_number']) for x in records}) == 92
    assert set(x['run_id'] for x in records) == {'phase2'}
    assert Counter(x['correctness_relation'] for x in records) == {
        'direct_fresh_original': 65, 'linked_to_first_output': 25, 'original_workload_oracle': 2}
    ids = {x['id']: x for x in records}
    for x in records:
        r = x['record']
        assert isinstance(r['wall_seconds'], (float, int)) and r['wall_seconds'] > 0
        assert x['target_count'] in (18141, 1024)
        assert x['experiment_count'] in (1, 3, 32)
        calls = r.get('estimator_calls', r.get('counters', r.get('diagnostics', r.get('worker_event', {}).get('counters', {}))).get('estimator_calls'))
        assert calls == x['target_count'], (x['id'], calls)
        if x['correctness_relation'] == 'direct_fresh_original':
            c = r['correctness']
            for k in ('exact_frame_equal', 'experiment_order_exact', 'invalid_masks_match', 'schema_and_dtypes_exact', 'serialized_csv_bytes_exact', 'target_order_exact', 'within_declared_tolerances'):
                assert c[k] is True, (x['id'], k)
            assert c['max_abs_numeric_difference'] == c['max_relative_error'] == 0
            assert c['declared_atol'] == 1e-8 and c['declared_rtol'] == 1e-7
            assert c['nan_locations'] == c['negative_inf_locations'] == c['positive_inf_locations'] == []
        elif x['correctness_relation'] == 'linked_to_first_output':
            first = ids[x['accepted_first_output_id']]
            assert first['correctness_relation'] == 'direct_fresh_original'
            assert first['record']['arm'] == r['arm'] and first['record']['trial'] == 0
            assert r['exact_to_first_request'] is True
            assert first['record']['csv_sha256'] == r['csv_sha256']
    groups = defaultdict(list)
    for x in records:
        groups[(x['group'], x['workload'], x['record']['arm'])].append(x)
    for (group, workload, arm), rows in groups.items():
        assert len(rows) == (3 if group == 'process_first_output' else 1 if group == 'pilot_oracle' else 5)
        if group in ('matched_resident', 'chunk_resident', 'pilot_resident'):
            assert sorted(x['record']['repetition'] for x in rows) == list(range(5))
        if group == 'process_first_output':
            assert sorted(x['record']['trial'] for x in rows) == list(range(3))
        if group == 'chunk_resident':
            for x in rows:
                r = x['record']
                assert r['diagnostics']['ffi_calls'] == r['expected_ffi_calls'] == math.ceil(x['target_count'] / r['chunk_size'])
    assert Counter(x['group'] for x in records) == {
        'chunk_resident': 15, 'matched_resident': 10, 'standalone_resident': 25,
        'process_first_output': 15, 'original_request': 5, 'pilot_oracle': 2, 'pilot_resident': 20}
    assert len(manifest['redactions']) == 50
    return groups

def summarize(records, data_dir=DATA):
    groups = validate(records, data_dir)
    rows = []
    for (group, workload, arm), members in sorted(groups.items()):
        row = {'group': group, 'workload': workload, 'arm': arm, 'units': 'seconds',
               'target_count': members[0]['target_count'], 'experiment_count': members[0]['experiment_count'],
               'record_ids': [x['id'] for x in members], **stats([x['record']['wall_seconds'] for x in members])}
        rows.append(row)
    def get(group, arm):
        return next(x for x in rows if x['group'] == group and x['arm'] == arm and x['target_count'] == 18141)
    memory = []
    for arm in PACKAGED:
        members = groups[('standalone_resident', 'original_three_experiments', arm)]
        maximum = max(x['record']['memory']['sampled_peak_rss_bytes'] for x in members)
        memory.append({'arm': arm, 'value_bytes': maximum, 'value_gib': maximum / (2**30),
                       'reduction': 'maximum of five standalone resident sampled RSS peaks; one value, no distribution',
                       'record_ids': [x['id'] for x in members],
                       'maximum_record_ids': [x['id'] for x in members if x['record']['memory']['sampled_peak_rss_bytes'] == maximum]})
    conversion = json.loads((data_dir / 'context.json').read_text())['records']['bundle_conversion']['data']['parent_launch_to_exit_seconds']
    def crossover(arm, extra=0):
        base_start, base_warm = get('process_first_output','source_python_fast')['median'], get('standalone_resident','source_python_fast')['median']
        candidate_start, candidate_warm = get('process_first_output', arm)['median'] + extra, get('standalone_resident', arm)['median']
        n = 1
        while candidate_start + (n - 1) * candidate_warm >= base_start + (n - 1) * base_warm:
            n += 1
            assert n < 100000
        return n
    return {'schema_version': 1, 'run_id': 'phase2', 'data_sha256': digest(data_dir / 'phase2_timings.jsonl'),
            'record_count': len(records), 'correctness': dict(Counter(x['correctness_relation'] for x in records)),
            'quartile_definition': 'linear interpolation at (n - 1) * q; IQR = q3 - q1',
            'groups': rows, 'standalone_memory': memory,
            'ratios': {'matched_B3_over_C0': get('matched_resident','B3')['median'] / get('matched_resident','C0')['median'],
                       'packaged_python_over_packed_load': get('standalone_resident','source_python_fast')['median'] / get('standalone_resident','packed_load')['median']},
            'bundle_conversion_seconds': conversion,
            'calculated_crossovers': {'label': 'Calculated scenario, not measured request sequences',
                                      'formula': 'first_output_median + (N - 1) * resident_median + optional_one_bundle_conversion',
                                      'packed_load_existing_bundle': crossover('packed_load'),
                                      'packed_load_with_conversion': crossover('packed_load', conversion),
                                      'source_rust': crossover('source_rust')}}

def write_tables(summary, records, destination):
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    fields = ['group','workload','arm','target_count','experiment_count','n','median','min','max','q1','q3','iqr','units','record_ids']
    with (destination / 'group_summary.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in summary['groups']:
            writer.writerow({**row,'record_ids': ';'.join(row['record_ids'])})
    fields=['id','run_id','group','workload','arm','target_count','experiment_count','repetition_or_trial','wall_seconds','correctness_relation','source','record_number']
    with (destination / 'observations.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for x in records:
            r=x['record']; row={k:x[k] for k in fields if k in x}
            row.update(arm=r['arm'],repetition_or_trial=r.get('repetition',r.get('trial','')),wall_seconds=r['wall_seconds'])
            writer.writerow(row)
    body=['# Generated evidence tables','','All values come from committed per-sample records. Time is wall seconds. Quartiles use linear interpolation.','','| Boundary | Workload | Arm | n | Median s | Min s | Max s | IQR s |','|---|---|---|---:|---:|---:|---:|---:|']
    for x in summary['groups']:
        body.append('| {group} | {workload} | {arm} | {n} | {median:.6f} | {min:.6f} | {max:.6f} | {iqr:.6f} |'.format(**x))
    body += ['', '## Standalone serving memory', '', 'Maximum of five sampled resident-request RSS peaks per standalone process; GiB = bytes / 2^30. No error bars or distribution are inferred.', '', '| Arm | Peak GiB |', '|---|---:|']
    body += ['| {arm} | {value_gib:.6f} |'.format(**x) for x in summary['standalone_memory']]
    body += ['', '## Comparisons', '', '- Matched B3 / C0 ratio of medians: %.6f×.' % summary['ratios']['matched_B3_over_C0'], '- Packaged Python / packed-load ratio of resident medians: %.6f×.' % summary['ratios']['packaged_python_over_packed_load'], '- One-time bundle conversion: %.9f seconds.' % summary['bundle_conversion_seconds'], '', 'Correctness: 65 direct fresh-original comparisons, 25 resident records linked to accepted first outputs, and 2 original workload-oracle records. Historical model-dependent tests were not rerun during publication.', '']
    (destination / 'summary.md').write_text('\n'.join(body))

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'figures/source_tables');p.add_argument('--check',action='store_true');args=p.parse_args()
    records=read_records();summary=summarize(records)
    if not args.check:write_tables(summary,records,args.out)
    print(json.dumps({'records':len(records),'correctness':summary['correctness'],'ratios':summary['ratios']},sort_keys=True))
if __name__=='__main__':main()
