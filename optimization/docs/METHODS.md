# Methods

## Scope and controls

The tested upstream is `pritchardlabatpsu/goloco` commit `470477c4be6ba0095016a2089b9bbf33d49a1ef8`. This contribution starts from that commit. The separately checked upstream HEAD at publication (2026-10-05) was also `470477c4be6ba0095016a2089b9bbf33d49a1ef8`; no untested upstream update was incorporated. The completed Phase 2 experiment used the authors' released L200 models and `data/PC9SKMELCOLO_L200_CERES_Features.csv`: 200 feature rows, experiment columns `ACH-000030`, `ACH-000615`, `ACH-001042`, and 18,141 supported targets. Each full request computes 54,423 gene-effect predictions and corresponding z-scores. The final table preserves 10 data columns, RangeIndex, and the original CSV index column. No training, Chronos fitting, sequencing processing, or dashboard deployment is included.

All measured work used one dedicated G4 allocation, CPU inference on AMD EPYC 9B45 (24 physical/48 logical CPUs, approximately 176.89 GiB RAM). The allocated GPU was unused. There was one compute worker, and `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `MKL_NUM_THREADS` were one. The isolated environment was Python 3.8.20, scikit-learn 0.22.1, NumPy 1.23.4, pandas 1.3.5, SciPy 1.9.3, and joblib 0.16.0; the [inference lock](../environments/inference.lock.txt) preserves the complete freeze independently of plotting dependencies.

The fitted models are single-output `RandomForestRegressor` objects containing `DecisionTreeRegressor` estimators. Effective feature widths are 1–10 and tree counts 53–110. Parameters, feature ordering, and tree counts are not modified to improve timing. Each request performs new inference; saved outputs serve only as equality references after timing.

## Implementation labels

| Label | Meaning |
| --- | --- |
| A | Original author inference and output construction; source model loading and feature-file parsing repeated within each request. |
| B2 | Earlier NumPy indexing and persistent model/metadata cache, using original estimator `.predict`; conceptual ancestry, not a new Phase 2 headline comparison. |
| B3 | Indexed request preparation and the same resident source models; original forest validation followed by serial calls to unchanged scikit-learn native trees, avoiding the forest's serial scheduling wrappers. |
| C0 | Matched B3 model/metadata residency plus native arrays; original validation followed by Rust forest traversal. This is the earlier Rust kernel reproduced on the Phase 2 allocation. |
| Packaged Python-fast | Independent `HeadlessGoloco`, with lifecycle, identity, cache-accounting, and ownership guards, using the B3 prediction behavior. |
| Packaged source Rust | Same independent API, loading source estimators and exporting native arrays within first use. |
| Packaged load / mmap | Same API, reading an existing verified numeric bundle through ordinary load or read-only memory mapping; original source estimators still loaded for validation. |

The reusable package imports no benchmark driver. Its three explicit backend choices are `numpy`, `python_fast`, and `rust`. Every request reconstructs feature-row indexing from current input. Selection preserves author membership semantics and input-row order, duplicates, experiment-column order, repeated targets, and original selected-matrix validation. There is no global validation step for unused feature values. Z-scores and final output assembly retain the original arithmetic and schema.

C0 reads the legacy validator's float32 input with signed strides, compares unchanged float64 thresholds using `<=`, accumulates leaf values in stored tree order, and divides once by the actual tree count. No fast-math or reassociation flags were enabled. Rust 1.99.0 used release opt-level 3, thin LTO, one code-generation unit, one build job, and `-C target-cpu=native`. The optional batch ABI calls this same per-target core. The core/profile functions were byte-identical across the initial and optional builds.

## Timing boundaries and sampling

Historical A and matched B3/C0 wall times run from input CSV parsing to output CSV close/flush in an imported process. A includes repeated source loading. B3/C0 share already prepared model/metadata residency; common loading and native export are recorded separately. A is therefore excluded from the equal-residency figure. Five B3/C0 pairs use the saved alternating balanced order seeded by 20261005. Untimed prevalidation and instrumented profiles are excluded from headline samples.

Fresh-process startup begins in the parent immediately before process launch and ends when it receives the event emitted immediately after the first CSV closes. It includes process startup, imports, source loading, verification, native export when needed, bundle checks/loading/page touching, prediction, and serialization. No candidate correctness prediction precedes that event. Three fresh workers were measured per variant. Downloads, environment setup, native compilation, and once-per-bundle conversion are separate. OS page caches were not controlled; these are fresh-process rather than cold-storage timings. All CSV sinks close/flush without durable `fsync`.

Only the first startup worker per variant performs five subsequent resident requests. Each resident clock includes CSV read, request-local selection, selected-input validation, fresh prediction, z-scores, table assembly, and CSV write/close. Standalone packaged observations are not pooled with minimal matched B3/C0 timings. The chunk comparison holds model/bundle residency constant and records five balanced repetitions for each of chunks 1, 64, and 256 in its own shared process.

Medians are computed from unrounded records; speedup annotations are ratios of medians. IQR uses linear quartiles. Plots show every timing sample and a median. Standalone serving memory is the maximum observed resident peak for each packaged worker, converted using 1 GiB = 2^30 bytes; it is neither a distribution nor the shared paired-process peak. The 16 GiB model/adapter cache cap is conservative capacity accounting, a different measure from RSS.

## Cache, bundle, and ownership

The LRU cache charges original models, feature metadata, adapters, owned native arrays, complete bundle storage, and source-identity bookkeeping. Full bundle bytes receive the same budget charge for load and mmap. The original-model allowance is the larger of twice serialized size and 64 KiB. An independent RSS limit is at most 60% of available RAM at construction. `prepare(targets)` is an explicit startup cost, not free work excluded from first use.

The deterministic converter verifies trusted source bytes, serializes six numeric NPY shards with a manifest and completion marker, verifies them, and publishes by atomic rename without overwriting a completed destination. Every open checks checksums, format/ABI, dimensions, offsets, roots, children, and tree structure. Mapping views retain their owners. Closing an engine releases its references without force-closing storage still owned by borrowed views. `invalidate(gene)` and `invalidate()` support explicit source/metadata reload; changed source objects incompatible with a bundle require an explicit rebuild. The instance rejects overlapping prediction, preparation, invalidation, close, and reopen operations.

## Correctness and evidence boundaries

The 92 timed records consist of 65 records with direct fresh-original comparisons, 25 resident records compared exactly to their accepted first output, and two original workload-oracle runs. The direct set comprises five A requests, ten B3/C0 requests, 15 startup outputs, 15 chunk outputs, and 20 one/32-column pilot outputs. This is not 92 independently recomputed original comparisons.

Every completed direct candidate comparison was exact for values, z-scores, target and experiment order, index, schema, dtype, signed nonfinite masks, and serialized CSV bytes. Maximum absolute and relative errors were zero; fixed `atol=1e-8, rtol=1e-7` checks were recorded separately. All 18,141 selected matrices received a separate position/order/shape/dtype/value audit. A secondary comparison with the authors' saved CSV was tolerance-equivalent but not bit-exact (maximum absolute difference 2.220446049250313e-16); fresh-original outputs define candidate acceptance.

The [compact historical validation evidence](../benchmarks/data/validation_summary.json) preserves these source hashes, counts, and pass flags. Historical regression coverage comprises 18 bundle tests, 120 backend contract checks, 45 lifecycle/state checks, 15 supplemental cases, 128 valid and 112 invalid native-boundary cases, and 38 chunk contract comparisons. Native probes covered 1,530 reached root thresholds and 7,650 input rows across 16 forests. Corruption, truncation, incompatible bundles, fresh-process reuse, source changes on copies, invalidation, eviction, held views, close/reopen, signed strides, unavailable Rust, and single-caller rejection were exercised. These counts describe completed Colab checks; publication source checks do not imply rerunning them.

The one/32-experiment pilots use the existing 1,024-target list. The one-column input is real; 32 columns are deterministic synthetic performance perturbations. The full-target expansion was skipped because the pilot's projected runtime exceeded the remaining experiment budget. [Limitations](LIMITATIONS.md) records the resulting scope.

The raw export and source-to-public redaction map are in [benchmark provenance](../benchmarks/PROVENANCE.md). [Figure provenance](../figures/README.md) links records, filters, hashes, boundaries, formulas, and units. Code and evidence were developed and audited using automated assistance; no human peer review or author endorsement is claimed.
