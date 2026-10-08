# Results

The packaged Rust backend was 4.26× faster than Python-fast for repeated requests. Python-fast reached the first output sooner.

## Startup and repeated requests

Workload: 18,141 targets, three original experiment columns, 54,423 predictions plus z-scores. First output runs from process launch until the first CSV closes. Resident requests run from CSV read to CSV close in the same worker.

| Backend | First output median, n=3 (s) | Resident median, n=5 (s) | Standalone serving peak (GiB) |
| --- | ---: | ---: | ---: |
| Python-fast | 56.618331 | 5.777371 | 7.428 |
| Rust, source export | 100.880155 | 1.436388 | 10.802 |
| Rust, packed ordinary load | 62.064298 | 1.354753 | 10.845 |
| Rust, packed mmap | 63.328703 | 1.416301 | 10.862 |

The 4.26× comparison includes the API's identity checks, cache accounting, and ownership checks. Ordinary load was slightly faster than mmap; both retain source estimators. Memory figures are maximum resident-request peaks, separate from the cache budget.

An existing bundle saves parameter export, but startup still loads source models and verifies checksums and structure. OS caches were uncontrolled. Downloads, builds, environment setup, and conversion are separate costs.

[Resident performance](../figures/reusable_resident.svg) · [PDF](../figures/reusable_resident.pdf) · [PNG](../figures/reusable_resident.png)

[Startup latency](../figures/startup_latency.svg) · [PDF](../figures/startup_latency.pdf) · [PNG](../figures/startup_latency.png)

[Standalone memory](../figures/standalone_memory.svg) · [PDF](../figures/standalone_memory.pdf) · [PNG](../figures/standalone_memory.png)

## Matched model residency

Five balanced pairs of equally resident Python (B3) and the earlier Rust kernel (C0) gave medians of 6.366502 s and 0.929296 s: a 6.85× ratio of medians. This is a replication, not an additional improvement over the previous Rust version. Common model/metadata loading and native export were timed separately.

The original implementation (A) took 65.172071 s, median of five requests. That clock includes repeated model loading and feature parsing, so A is excluded from the matched plot. All timings come from the same Phase 2 allocation.

[Matched residency](../figures/matched_residency.svg) · [PDF](../figures/matched_residency.pdf) · [PNG](../figures/matched_residency.png)

## When startup pays off

Bundle conversion, serialization, and verification took 99.145200 s by the parent timer. Six numeric shards occupy 3,589,582,656 bytes (3.343 GiB). All three startup trials per bundle mode and their resident requests performed zero native parameter exports. Original estimators remain necessary for input validation.

Using `first_output_median + (N - 1) * resident_median`, ordinary-load Rust overtakes Python-fast at request 3 with an existing bundle, or request 25 including conversion. Source-export Rust crosses at request 12. These are calculated crossover counts, not measured long sequences. Fresh processes for every request never cross over at these costs. Native compilation and toolchain setup remain separate. The fork does not include the bundle.

## Chunking made requests slower

| Targets per native call | Resident median, n=5 (s) | Actual FFI calls per full request |
| --- | ---: | ---: |
| 1 | 1.438062 | 18,141 |
| 64 | 5.068635 | 284 |
| 256 | 5.053049 | 71 |

All outputs were exact. Batching reduced calls but increased latency, so the core API keeps `chunk_size=1`. The experimental class and runner remain available; the diagnostics do not explain the full slowdown.

[Chunking result](../figures/negative_chunking.svg) · [PDF](../figures/negative_chunking.pdf) · [PNG](../figures/negative_chunking.png)

## Correctness and tested scope

The 92 timed records contain 65 direct fresh-original comparisons, 25 resident requests checked against accepted first outputs, and two original pilot-oracle runs. Completed candidate comparisons were exact for predictions, z-scores, feature selection, target and experiment order, schema/dtype, nonfinite masks, and CSV bytes. Direct absolute and relative errors were zero; tolerance checks were separate. The authors' saved CSV was tolerance-equivalent but not bit-exact. Acceptance uses the fresh original.

The 18,141-target matrix audit and historical regression tests passed. [Methods](METHODS.md) gives test coverage and counts, including direct versus transitive checks.

The 1,024-target pilots completed with one real column and 32 synthetic performance columns. Their full-target expansions were not run. Synthetic columns contain no new biological measurements.

Source data: [per-sample records](../benchmarks/data/phase2_timings.jsonl), [export provenance](../benchmarks/PROVENANCE.md), [figure captions](../figures/CAPTIONS.md), and [summary tables](../figures/source_tables/summary.md).
