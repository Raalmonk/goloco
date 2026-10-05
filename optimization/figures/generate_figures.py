#!/usr/bin/env python3
"""Regenerate five figures and tables from committed sanitized evidence only."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('goloco_evidence_summary', ROOT / 'benchmarks/summarize.py')
evidence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evidence)
PACKAGED = evidence.PACKAGED
LABELS = {'B3': 'B3 · Python tree loop', 'C0': 'C0 · Rust traversal',
          'source_python_fast': 'Python-fast', 'source_rust': 'Rust · source export',
          'packed_load': 'Rust · bundle load', 'packed_mmap': 'Rust · bundle mmap',
          'chunk1': 'Chunk 1', 'chunk64': 'Chunk 64', 'chunk256': 'Chunk 256'}
COLORS = {'source_python_fast': '#0072B2', 'B3': '#0072B2'}

plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11, 'axes.titlesize':16,
    'axes.labelsize':11, 'xtick.labelsize':10, 'ytick.labelsize':11,
    'svg.fonttype':'none', 'svg.hashsalt':'goloco-phase2-publication-v1',
    'pdf.fonttype':42, 'ps.fonttype':42, 'axes.spines.top':False,
    'axes.spines.right':False, 'axes.spines.left':False, 'figure.facecolor':'white',
    'axes.facecolor':'white', 'axes.axisbelow':True, 'savefig.facecolor':'white'})

def export(fig, out, stem):
    # Omit wall-clock metadata so pinned-environment repeated renders are byte-identical.
    for ext in ('svg','pdf','png'):
        metadata = {'Date':None} if ext == 'svg' else ({'CreationDate':None,'ModDate':None} if ext == 'pdf' else {'Software':'GOLOCO evidence figures'})
        fig.savefig(out / (stem + '.' + ext), dpi=300, metadata=metadata)
    plt.close(fig)

def chart(records, group, arms, title, subtitle, out, stem, annotation=None, ffi=None, highlight=False):
    rows = [x for x in records if x['group']==group and x['record']['arm'] in arms]
    values = [[x['record']['wall_seconds'] for x in rows if x['record']['arm']==a] for a in arms]
    stats = [evidence.stats(x) for x in values]
    fig, ax = plt.subplots(figsize=(8.8, 4.1 if len(arms)==2 else 4.7))
    fig.subplots_adjust(left=.27,right=.94,top=.72,bottom=.22 if len(arms)==2 else .19)
    maximum=max(max(v) for v in values)
    if highlight and 'packed_load' in arms:
        ax.axhspan(arms.index('packed_load')-.38, arms.index('packed_load')+.38, color='#F3EEE6',zorder=0)
    markers=('o','s','^','D')
    for i,(arm,observations,s) in enumerate(zip(arms,values,stats)):
        offsets=np.linspace(-.16,.16,len(observations))
        color=COLORS.get(arm,'#B65C00')
        ax.scatter(observations,i+offsets,marker=markers[i%len(markers)],s=42,facecolors='white',edgecolors=color,linewidths=1.5,zorder=4)
        ax.plot([s['q1'],s['q3']],[i,i],color='black',lw=4,solid_capstyle='butt',zorder=3)
        ax.plot([s['median'],s['median']],[i-.24,i+.24],color='black',lw=1.8,zorder=5)
        ax.text(maximum*1.055,i,f"{s['median']:.3f} s",ha='left',va='center',fontsize=10)
    ax.set_xlim(0,maximum*1.23); ax.set_ylim(len(arms)-.55,-.60)
    labels=[LABELS[a] + (f"\n{ffi[a]:,} FFI calls" if ffi else '') for a in arms]
    ax.set_yticks(range(len(arms)),labels);ax.tick_params(axis='y',length=0,pad=12)
    ax.set_xlabel('Wall time (seconds)');ax.grid(axis='x',color='#DDDDDD',lw=.6)
    fig.text(.055,.93,title,fontsize=16,weight='bold',ha='left',va='top')
    fig.text(.055,.85,subtitle,fontsize=10.5,ha='left',va='top')
    if annotation:fig.text(.055,.785,annotation,fontsize=11,weight='bold',ha='left',va='top')
    handles=[Line2D([0],[0],marker='o',color='none',markeredgecolor='#555555',markerfacecolor='white',label='All observations'),Line2D([0],[0],color='black',lw=4,label='IQR'),Line2D([0],[0],marker='|',markersize=12,color='none',markeredgecolor='black',label='Median')]
    fig.legend(handles=handles,loc='lower left',bbox_to_anchor=(.265,.015),ncol=3,frameon=False,fontsize=9,columnspacing=1.2,handlelength=1.2)
    export(fig,out,stem)
    return rows

def memory_chart(summary,out):
    rows=summary['standalone_memory']; values=[x['value_gib'] for x in rows]
    fig,ax=plt.subplots(figsize=(8.8,4.7));fig.subplots_adjust(left=.27,right=.92,top=.72,bottom=.17)
    bars=ax.barh(range(4),values,height=.46,color=['#0072B2','#B65C00','#B65C00','#B65C00'],edgecolor='black',linewidth=.6)
    bars[2].set_hatch('//')
    for i,v in enumerate(values):ax.text(v+.18,i,f'{v:.3f} GiB',ha='left',va='center',fontsize=10)
    ax.set_yticks(range(4),[LABELS[a] for a in PACKAGED]);ax.tick_params(axis='y',length=0,pad=12)
    ax.set_ylim(3.6,-.6);ax.set_xlim(0,max(values)*1.24);ax.grid(axis='x',color='#DDDDDD',lw=.6)
    ax.set_xlabel('Sampled peak RSS (GiB)')
    fig.text(.055,.93,'Standalone serving memory',fontsize=16,weight='bold',va='top')
    fig.text(.055,.85,'18,141 targets × 3 experiments · one serving process per variant',fontsize=10.5,va='top')
    fig.text(.055,.785,'Maximum over five resident-request peaks; no inferred error bars',fontsize=10.5,va='top')
    fig.text(.27,.035,'GiB = bytes / 2³⁰; original source estimators remain resident.',fontsize=9)
    export(fig,out,'standalone_memory')

def provenance_entry(stem,rows,data_hash,boundary,formula,units,fields):
    return {'figure':stem,'data_file':'../benchmarks/data/phase2_timings.jsonl','data_sha256':data_hash,
            'record_ids':[x['id'] for x in rows], 'source_files':sorted(set(x['source'] for x in rows)),
            'filters':{'groups':sorted(set(x['group'] for x in rows)), 'arms':sorted(set(x['record']['arm'] for x in rows)), 'target_count':18141,'experiment_count':3,'run_id':'phase2'},
            'measurement_fields':fields,'timing_boundary':boundary,'formula':formula,'units':units}

def documentation(out,summary,selected,records):
    manifest=json.loads((ROOT/'benchmarks/data/export_manifest.json').read_text())
    conversion=summary['bundle_conversion_seconds'];ratio=summary['ratios']
    boundary='CSV parsing through closed/flushed original-schema CSV; no fsync; source models and adapters already resident; correctness outside timing'
    entries=[]
    captions={
    'matched_residency':('Matched-residency inference',f"Five paired observations for B3 (Python tree loop) and C0 (Rust traversal), with both adapters and common source models already resident. Each request covers 18,141 targets × the original three experiment columns. Open symbols show every observed wall time; thick horizontal intervals show the interquartile range (Q1–Q3); vertical ticks and labels show the median. Quartiles use linear interpolation at (n − 1) × q. The ratio of the unrounded medians is {ratio['matched_B3_over_C0']:.6f}×. This is a matched-residency reproduction of the earlier Rust comparison, not an additional improvement over a previous Rust version. Original A includes repeated loading and is deliberately excluded from this chart.",f"Five B3 observations have median {next(x['median'] for x in summary['groups'] if x['group']=='matched_resident' and x['arm']=='B3'):.3f} seconds; five C0 observations have median {next(x['median'] for x in summary['groups'] if x['group']=='matched_resident' and x['arm']=='C0'):.3f} seconds."),
    'reusable_resident':('Reusable backend resident performance',f"Five repeated requests per packaged variant in its own standalone worker after its accepted first output, using all 18,141 targets and three experiment columns. Every request parses the feature CSV, performs fresh selection, input validation and inference, constructs z-scores/table, and closes the output CSV. Symbols, IQR intervals and median ticks are defined as in Figure 1. Ordinary bundle loading at chunk size 1 is highlighted as the measured resident-workload choice. Its ratio versus packaged Python-fast is {ratio['packaged_python_over_packed_load']:.6f}×; this is separate from the minimal B3/C0 benchmark. Original estimators remain required in both bundle modes.",'Four standalone resident groups show Python-fast near 5.78 seconds and Rust modes near 1.35–1.44 seconds; all five observations in each group are visible.'),
    'startup_latency':('Launch to first completed output',f"Three fresh-process trials per packaged variant. Timing begins in the parent immediately before process launch and ends on receipt of the event emitted just after the first CSV is closed/flushed, without fsync. Imports, verification, source-model loading, required export/page touching, inference and CSV construction are included. No candidate correctness prediction precedes the event. Open symbols show all trials; intervals show linear Q1–Q3 and ticks show medians. OS caches were uncontrolled; this is not a cold-storage measurement. Bundle modes assume an existing bundle. Downloads, environment/native builds, and the separately measured {conversion:.9f}-second conversion/serialization/verification are excluded. Conversion is caption context, not a measured stacked timing segment.",'Python-fast starts in about 57 seconds, source-export Rust about 101 seconds, and existing-bundle Rust about 62–63 seconds; all three process trials are visible.'),
    'standalone_memory':('Standalone serving memory','Each bar is the maximum of the five sampled resident-request RSS peaks in the corresponding standalone worker, sampled every 10 ms including descendants and excluding the orchestration parent (start/end samples are also retained). These are four maxima, not four estimated distributions; no error bars are inferred. GiB = bytes / 2^30. This chart does not use the shared B3/C0 process RSS or cache-budget accounting. Both bundle modes still retain source estimators for the legacy validator. The hatched bar denotes ordinary bundle loading.','Standalone resident peak RSS is about 7.43 GiB for Python-fast and 10.80–10.86 GiB for Rust variants.'),
    'negative_chunking':('Negative chunking result','Five balanced-order resident requests for each chunk size, using the same prepared source-model and mmap-array state. All 18,141 targets and three experiments are evaluated on every request. Symbols, IQR intervals and median ticks are defined as in Figure 1. Actual recorded FFI calls are 18,141, 284 and 71 for chunks 1, 64 and 256. Fewer calls did not reduce end-to-end latency in this implementation: larger chunks were slower while all completed comparisons remained exact. The complete slowdown has not been causally assigned to any single stage. Keep chunk size 1 as the measured default.','Chunk 1 takes about 1.44 seconds despite 18,141 calls; chunks 64 and 256 take about 5.05–5.07 seconds with only 284 and 71 calls.')}
    for stem,rows in selected.items():
        ismem=stem=='standalone_memory'; startup=stem=='startup_latency'
        fields=['record.memory.sampled_peak_rss_bytes'] if ismem else ['record.wall_seconds']
        if stem=='negative_chunking':fields+=['record.diagnostics.ffi_calls']
        formula='maximum(sampled_peak_rss_bytes) / 2^30 per standalone arm' if ismem else 'all observations; median; Q1 and Q3 by linear interpolation at (n-1)*q'
        if stem in ('matched_residency','reusable_resident'):formula+='; ratio of unrounded arm medians'
        e=provenance_entry(stem,rows,summary['data_sha256'],'per-resident-request sampled RSS; maximum over five requests' if ismem else ('parent before Popen through first CSV-close event receipt; no fsync' if startup else boundary),formula,'GiB' if ismem else 'seconds',fields)
        e['original_source_sha256']={s:manifest['source_files'][s]['sha256'] for s in e['source_files']}
        entries.append(e)
    (out/'provenance.json').write_text(json.dumps({'schema_version':1,'source_consolidated_sha256':manifest['source_sha256'],'figures':entries},indent=2,sort_keys=True)+'\n')
    text=['# Figure captions and alt text','','All figures are generated from one accepted Phase 2 allocation. CPU inference used one compute worker on AMD EPYC 9B45 with BLAS/OpenMP thread counts fixed to one; the allocated GPU was unused. No new inference was performed for this publication.']
    for i,stem in enumerate(selected,1):
        title,caption,alt=captions[stem];e=next(x for x in entries if x['figure']==stem)
        text+=['',f'## Figure {i}. {title}','',f'![{alt}]({stem}.png)','',f'**Caption.** {caption}','',f'**Alt text.** {alt}','',f'[SVG]({stem}.svg) · [PDF]({stem}.pdf) · [300-dpi PNG]({stem}.png)','',f"**Provenance.** [`phase2_timings.jsonl`](../benchmarks/data/phase2_timings.jsonl), records {', '.join(e['record_ids'])}. Filter: run `phase2`, group `{e['filters']['groups'][0]}`, arms `{', '.join(e['filters']['arms'])}`, 18,141 targets, three experiments. Field(s): `{', '.join(e['measurement_fields'])}`. Boundary: {e['timing_boundary']}. Formula: {e['formula']}. Units: {e['units']}. Public source SHA-256: `{summary['data_sha256']}`. Per-original-file hashes are in [`provenance.json`](provenance.json)."]
    text+=['','Conversion context comes from `context.json#/records/bundle_conversion`, whose original receipt hash and retained-field map are included in that public file. It is not an inference sample.','']
    (out/'CAPTIONS.md').write_text('\n'.join(text))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parent);args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    records=evidence.read_records();summary=evidence.summarize(records)
    evidence.write_tables(summary,records,args.out/'source_tables')
    selected={}
    selected['matched_residency']=chart(records,'matched_resident',('B3','C0'),'Matched-residency inference','18,141 targets × 3 experiments · n = 5 paired observations per arm',args.out,'matched_residency',annotation=f"Ratio of medians: {summary['ratios']['matched_B3_over_C0']:.2f}×")
    selected['reusable_resident']=chart(records,'standalone_resident',PACKAGED,'Reusable backend resident performance','18,141 targets × 3 experiments · n = 5 requests per standalone arm',args.out,'reusable_resident',annotation=f"Python-fast / bundle load: {summary['ratios']['packaged_python_over_packed_load']:.2f}×",highlight=True)
    selected['startup_latency']=chart(records,'process_first_output',PACKAGED,'Launch to first completed output','18,141 targets × 3 experiments · n = 3 fresh processes per arm',args.out,'startup_latency',annotation='Existing bundle assumed · OS cache state uncontrolled')
    memory_chart(summary,args.out)
    selected['standalone_memory']=[x for x in records if x['group']=='standalone_resident' and x['record']['arm'] in PACKAGED]
    chunkarms=('chunk1','chunk64','chunk256')
    ffi={arm:next(x['record']['diagnostics']['ffi_calls'] for x in records if x['group']=='chunk_resident' and x['record']['arm']==arm) for arm in chunkarms}
    selected['negative_chunking']=chart(records,'chunk_resident',chunkarms,'Fewer FFI calls did not reduce latency','18,141 targets × 3 experiments · n = 5 requests per chunk size',args.out,'negative_chunking',annotation='Completed negative result · all comparisons exact',ffi=ffi)
    documentation(args.out,summary,selected,records)
    print(json.dumps({'figures':list(selected),'formats':['svg','pdf','png'],'records':len(records),'source_sha256':summary['data_sha256']},sort_keys=True))
if __name__=='__main__':main()
