# Data and artifact availability

This fork contains the accepted reusable Python package, Rust source, bundle and contract test sources, historical benchmark drivers, target lists, compact sanitized measurements, descriptive summaries, figure-generation code, and SVG/PDF/PNG figures. The authors' original repository content and notices are retained.

The [public timing records](../benchmarks/data/phase2_timings.jsonl) preserve all 92 Phase 2 timed records and their grouping, units, and correctness relations. The [compact validation summary](../benchmarks/data/validation_summary.json) preserves historical test counts and pass flags with source hashes and an explicit field-selection map. The [export map](../benchmarks/PROVENANCE.md) binds local source hashes to public derivatives and describes redactions. [SCIENTIFIC_SOURCE_MANIFEST.json](../SCIENTIFIC_SOURCE_MANIFEST.json) binds the scientific modules to accepted source hashes. Figure values are regenerated from these records, not from manually entered medians. No parent-phase measurements are pooled into the Phase 2 figures.

Retrieve author models from the [official Zenodo record](https://zenodo.org/records/14251390), archive `ceres-infer.zip`. The exact archive used here has:

| Property | Recorded value |
| --- | --- |
| Bytes | 5,130,807,234 |
| MD5 | `3811e49729e22d3146d2d57c04f0834c` |
| SHA-256 | `e02feb3118663982d74c9e8187dd47c5757686b6b3bcc59d090ba4fd6d8ad30e` |
| L200 source files | 36,282: 18,141 model pickles and 18,141 feature CSVs |

[model_source_identity.json](../evidence/model_source_identity.json) also provides a deterministic compact inventory digest. The [preparation helper](../scripts/prepare_models.py) verifies the complete official archive, extracts only L200 model/feature files, verifies that inventory, and produces the per-file manifest required by the converter. It never unpickles files. Prediction and conversion subsequently load trusted original model pickles; checksums identify expected bytes but do not make arbitrary pickle input safe.

The native bundle is **not publicly downloadable from this fork**. Its converter is supplied. The measured six numeric shards contain 3,589,582,656 bytes (approximately 3.343 GiB), in addition to metadata and completion records. Source estimators remain necessary even after conversion because the accepted backend retains their legacy validator. [Reproduction instructions](REPRODUCE.md) show retrieval and conversion separately from figure generation.

Complete original evidence, author model archives when retained, native bundle and TAR, private operational logs, and session receipts remain in the owner's preserved local experiment directories. They were not placed in Git. This contribution also excludes environments, Cargo build outputs, compiled native binaries, credentials, session information, private prompts, and conversation transcripts. It does not redistribute model files or claim new rights over author models/data; consult the official source and [attribution record](ATTRIBUTION.md).
