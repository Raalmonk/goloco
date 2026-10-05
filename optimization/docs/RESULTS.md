# Results

The reusable backend improves repeated-request CPU inference while retaining exact outputs on the executed workload. It does not improve first-use latency over optimized Python. All measurements below are from the completed Phase 2 allocation; the parent experiment supplies implementation ancestry rather than pooled timing samples.

## Matched model residency

Five balanced B3/C0 pairs at full scope gave medians of **6.366502 s** and **0.929296 s**, a **6.85× ratio of medians**. This remeasures the existing C0 Rust kernel against equally resident B3 Python on one allocation. It is not a further gain over the previous Rust version. Common source/metadata loading and native export were excluded from these resident clocks and measured separately.

The original A request median was 65.172071 s (five observations). Its clock includes repeated model loading and feature parsing, so it is not plotted as an equally resident control.

[Figure 1 — matched residency](../figures/matched_residency.svg) · [PDF](../figures/matched_residency.pdf) · [PNG](../figures/matched_residency.png)

## Packaged startup and repeated requests

The full workload is 18,141 targets × three original experiment columns (54,423 predictions plus z-scores). First output is parent-observed process launch through first completed CSV; resident requests include CSV read through CSV close in the existing worker.

| Backend | First output median, n=3 (s) | Resident median, n=5 (s) | Standalone serving peak (GiB) |
| --- | ---: | ---: | ---: |
| Python-fast | 56.618331 | 5.777371 | 7.428 |
| Rust, source export | 100.880155 | 1.436388 | 10.802 |
| Rust, packed ordinary load | 62.064298 | 1.354753 | 10.845 |
| Rust, packed mmap | 63.328703 | 1.416301 | 10.862 |

Packaged Python-fast versus ordinary-load Rust is a separate **4.26× resident comparison**, including the reusable API's identity, cache, and ownership bookkeeping. Ordinary load was slightly faster than mmap in this workload; both retain source estimators and have similar serving RSS. Peak memory is a measured maximum of the resident request samples, not a distribution or cache-cap estimate.

For one request per fresh process, Python-fast wins: 56.618 s versus 62.064 s for an existing ordinary-load bundle. Source-export Rust costs 100.880 s at first use. An existing bundle avoids parameter export but still requires full checksum/structure verification and original-model loading. OS caches were uncontrolled. Downloads, builds, environment setup, and conversion are separate from first-output timing.

[Figure 2 — reusable resident performance](../figures/reusable_resident.svg) · [PDF](../figures/reusable_resident.pdf) · [PNG](../figures/reusable_resident.png)  
[Figure 3 — startup latency](../figures/startup_latency.svg) · [PDF](../figures/startup_latency.pdf) · [PNG](../figures/startup_latency.png)  
[Figure 4 — standalone memory](../figures/standalone_memory.svg) · [PDF](../figures/standalone_memory.pdf) · [PNG](../figures/standalone_memory.png)

## Bundle cost and calculated service scenarios

One-time bundle conversion, serialization, and verification took **99.145200 s** by the parent timer. Six numeric shards contain 3,589,582,656 bytes (3.343 GiB); the native bundle is not distributed in this fork. All three first-output trials for each bundle mode and their resident requests recorded zero native parameter exports. Original estimators remain required for input validation.

A calculated scenario uses `first_output_median + (N - 1) * resident_median` within this same allocation. Ordinary-load Rust first beats Python-fast at request 3 if a bundle already exists, and at request 25 when charging one measured bundle conversion. Source-export Rust crosses at request 12. These are calculated crossover counts, not measured long request sequences. Repeated independent one-request processes do not cross over under these median costs. Native compilation and toolchain setup remain separate costs.

## Chunking: completed negative result

| Targets per native call | Resident median, n=5 (s) | Actual FFI calls per full request |
| --- | ---: | ---: |
| 1 | 1.438062 | 18,141 |
| 64 | 5.068635 | 284 |
| 256 | 5.053049 | 71 |

All chunk comparisons passed exactness checks. Fewer calls did not reduce latency in this implementation, so the core API retains `chunk_size=1`. The separate experimental class and runner are preserved. The available diagnostics do not attribute the entire slowdown to a single cause.

[Figure 5 — negative chunking result](../figures/negative_chunking.svg) · [PDF](../figures/negative_chunking.pdf) · [PNG](../figures/negative_chunking.png)

## Correctness and scope

The saved corpus has 92 timed records: 65 direct fresh-original comparisons, 25 resident requests linked to accepted first outputs, and two original pilot-oracle runs. All completed candidate comparisons were exact for predictions, z-scores, selection, target/experiment order, schema/dtype, nonfinite masks, and CSV bytes. Direct comparisons report zero absolute and relative error. Fixed tolerance checks are separate. The authors' saved CSV fixture is tolerance-equivalent but not bit-exact; candidate acceptance uses the fresh original.

The full 18,141-target matrix audit passed. Historical tests passed for bundle validation, backend contracts, lifecycle and cache behavior, boundary probes, and chunk ownership/error behavior. [Methods](METHODS.md) gives counts and distinguishes direct from transitive evidence. The 1,024-target one-real-column and 32-synthetic-column pilots completed; expanded one/32-column full-target workloads were not run. The synthetic columns are performance data, not new biology.

All plotted values and generated tables come from [sanitized per-sample records](../benchmarks/data/phase2_timings.jsonl), with [export provenance](../benchmarks/PROVENANCE.md), [figure captions](../figures/CAPTIONS.md), and [recomputed summary tables](../figures/source_tables/summary.md). No new inference, model conversion, native compilation, or Colab allocation was performed for publication.
