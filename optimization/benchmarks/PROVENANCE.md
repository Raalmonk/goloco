# Evidence provenance and export map

The public data derives from the accepted Phase 2 experiment only. The tested upstream revision is `470477c4be6ba0095016a2089b9bbf33d49a1ef8`. Parent experiment data supplies historical lineage elsewhere in the documentation; its measurements are not pooled into these figures.

## Per-sample timing export

[`data/phase2_timings.jsonl`](data/phase2_timings.jsonl) retains all **92** original wrapped timing records. The original consolidation has `source`, `record_number` and `record`; values must be read from `record`. Public IDs `p2-001` through `p2-092` are the one-based original consolidated line numbers. The `source` values are relative scientific evidence labels, not live filesystem locations or session identifiers.

[`data/export_manifest.json`](data/export_manifest.json) records:

- The SHA-256 of the preserved original consolidated file and the public derivative.
- SHA-256 and sample counts for each original per-group JSONL source.
- Exact original source-line hashes, also retained in each public record, without the trailing newline.
- Every changed JSON pointer: **50** `record.cache_before.model_bundle` or `record.cache_after.model_bundle` strings were replaced by `artifact:native_bundle`. The private absolute artifact directory is not published.
- The added field map for public IDs, grouping, workload counts and correctness relationships.

No numeric measurement, counter, list, correctness flag or sample was removed, added or rounded. All other original record fields remain unchanged. Before export, each wrapped record was compared with the indicated original source JSONL line. Publication also independently compared the sanitized derivative against the local raw records. The source originals remain unchanged locally.

## Correctness accounting

| Class | Count | Evidence |
|---|---:|---|
| `direct_fresh_original` | 65 | Explicit per-record fresh-original frame, schema, ordering, nonfinite-mask, exact CSV and fixed-tolerance results |
| `linked_to_first_output` | 25 | Exact resident output and CSV hash matched to the same arm's accepted trial-0 first output; `accepted_first_output_id` retains this link |
| `original_workload_oracle` | 2 | Original one-column and synthetic 32-column pilot requests that created their reference output |

The 92 records are not 92 independently recomputed original comparisons. All completed candidate comparisons were exact in the executed scope; fixed `atol=1e-8`, `rtol=1e-7` flags are retained separately. Original A stability measurements, paired controls, package startup, resident requests, pilots and chunks have distinct groups.

## Grouping and boundaries

| Group | Records | Boundary and workload |
|---|---:|---|
| `original_request` | 5 | Original A, full targets × three experiments, imported-process CSV parse through CSV close including per-request loading |
| `matched_resident` | 10 | Five full-target B3/C0 pairs; both adapters and common models resident; CSV parse through CSV close |
| `process_first_output` | 15 | Three fresh-process trials × five variants; parent before launch through receipt immediately after first CSV close |
| `standalone_resident` | 25 | Five requests × five standalone variants after their accepted first output; CSV parse through CSV close |
| `chunk_resident` | 15 | Five full-target requests × chunks 1/64/256, common prepared mmap backend; CSV parse through CSV close |
| `pilot_oracle` | 2 | Original 1,024-target references, one original experiment or 32 synthetic performance columns |
| `pilot_resident` | 20 | Five B3/C0 pairs per 1,024-target pilot workload |

CSV close/flush does not include fsync. Correctness comparisons are outside measured inference intervals. OS page caches were uncontrolled. Bundle startup assumes a pre-existing bundle; conversion/build/download phases are separate. Setup, diagnostic profiles and untimed prevalidation are absent from this timing export and never contribute to headline inference medians. The synthetic 32-column workload is performance data, not newly collected biology. Full-target one/32-column expansion was not run.

Pair identity is `(run_id, group, workload, repetition)`. Paired B3/C0 and chunk groups contain one record per arm in every repetition 0–4. No fresh-process trials from different allocations are pooled. Standalone memory uses only `record.memory.sampled_peak_rss_bytes` from resident workers: one maximum over five requests per arm. Shared-process RSS, process high-water RSS and cache accounting are retained in raw records but are not substituted for that chart.

## Supplemental context

[`data/context.json`](data/context.json) contains allowlisted fields from raw conversion receipts, protocols and saved validation summaries. Each entry lists its exact source label, source SHA-256, retained fields and retained data. Omitted fields are operational commands, absolute paths, epochs, deadlines or unrelated large metadata. This supplemental context does not add inference samples. The conversion caption derives from the actual parent-timed receipt. Bundle metadata is read as JSON only; no bundle arrays are published or loaded by figure generation.

[`data/validation_summary.json`](data/validation_summary.json) separately preserves compact historical bundle, backend, lifecycle, supplemental, native-boundary, matrix and chunk validation evidence. Each entry retains its original source hash and extraction map. The unittest terminal count was extracted without publishing the raw operational log. Its public derivative hash is in `export_manifest.json`; these records do not add to the 92 timed observations. Model-dependent tests remain historical.

## Recalculation

[`summarize.py`](summarize.py) uses Python's standard library and validates record counts, group sizes, pairing, estimator-call counts, per-record correctness, resident-to-startup links and CSV hashes. It computes medians and linear-interpolated quartiles at `(n − 1) × q`; `IQR = Q3 − Q1`. Speed ratios divide unrounded medians. GiB divides measured bytes by `2^30`. Calculated crossovers use `first_output_median + (N − 1) × resident_median`, optionally plus one bundle conversion, and are labeled as calculated scenarios rather than measured request sequences.

[`../figures/provenance.json`](../figures/provenance.json) maps every chart to exact public record IDs, source files/hashes, fields, filters, boundaries, formulas and units. [`../figures/source_tables/observations.csv`](../figures/source_tables/observations.csv) and [`../figures/source_tables/group_summary.csv`](../figures/source_tables/group_summary.csv) are generated tables, not primary evidence. A fresh checkout needs only the small committed inputs and the separate plotting lock to regenerate them.
