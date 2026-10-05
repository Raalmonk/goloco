# Reusable GOLOCO CPU inference

This independent contribution preserves the authors' scientific implementation and adds a reusable inference backend, verified numeric model conversion, and reproducible performance evidence. It covers compressed CERES feature input through the original genome-wide prediction table. The upstream application and author documentation remain intact.

For the measured 18,141-target, three-experiment workload, packaged Python completed a first request in **56.62 s**, compared with **62.06 s** for Rust using an existing ordinary-load bundle. In the same persistent worker, resident medians were **5.78 s** and **1.35 s**, respectively (**4.26×**, five observations each). Use Python-fast for an isolated request under these measured conditions; a persistent Rust process amortizes its additional startup work.

![All five matched-residency observations: B3 median 6.37 seconds and C0 median 0.93 seconds, ratio 6.85.](figures/matched_residency.png)

The figure above is the minimal B3/C0 matched-residency comparison. It reproduces the previous Rust kernel against an equally resident Python control on the new allocation; **6.85× is not an additional gain over the earlier Rust version**. The packaged API has different bookkeeping and is reported separately. [Results](docs/RESULTS.md) · [Methods](docs/METHODS.md) · [Figures and captions](figures/README.md) · [Limitations](docs/LIMITATIONS.md) · [Attribution](docs/ATTRIBUTION.md)

## Regenerate figures without models

From the repository root, follow the single figure command in [figures/README.md](figures/README.md). Its separate lightweight environment reads only committed sanitized records and regenerates SVG, PDF, 300-dpi PNG, and summary tables. It does not import the inference package, download models, or allocate a runtime.

## Inference API

[Reproduction instructions](docs/REPRODUCE.md) describe the pinned Linux environment, trusted official model retrieval, native build, bundle conversion, CLI, and historical model-dependent tests. After that setup, run from the repository root with `PYTHONPATH=optimization`:

```python
from pathlib import Path
import json
import pandas as pd
from goloco_reusable import HeadlessGoloco

project = Path.cwd()
repo = project / "work/upstream"
model_root = Path("work/models")
frame = pd.read_csv(repo / "data/PC9SKMELCOLO_L200_CERES_Features.csv")
targets = json.loads((project / "optimization/evidence/canonical_targets.json").read_text())

with HeadlessGoloco(
    repo=repo,
    model_root=model_root,
    backend="rust",
    model_bundle=Path("work/native_bundle"),
    bundle_mode="load",
    library=project / "optimization/rust/target/release/libgoloco_forest.so",
    max_cache_bytes=16 * 1024**3,
    chunk_size=1,
) as engine:
    result = engine.predict(frame, targets)
    result.to_csv("work/prediction.csv")
    # Reuse this instance for subsequent current input values.
    print(engine.snapshot())
```

The explicit alternatives are `numpy` (original estimator prediction with indexed input preparation), `python_fast` (original validator and unchanged scikit-learn native trees in stored order), and `rust` (original validator and the accepted native traversal). Rust requires an explicit compatible compiled library and has no silent fallback. Source-start Rust omits `model_bundle`; it exports parameters on model-cache misses.

The documented `bundle_mode="load"`, `chunk_size=1` choice performed best among the packaged resident variants here. The preserved constructor and CLI still default to `mmap` when bundle mode is omitted; set `load` explicitly for this example. The experimental `ChunkedGoloco` class and [chunk comparison](experiments/chunk_comparison.py) remain available. Chunks 64 and 256 were correct but slower, despite fewer native calls.

The original source estimators remain required for selected-input validation in every backend, including bundle serving. Persistent state contains models, metadata, identities, and immutable native arrays; it never contains saved final predictions or current input values. The instance is single-caller and rejects overlapping operations. [Data availability](docs/DATA_AVAILABILITY.md) explains what is published and what must be retrieved separately.

## Evidence and checks

All model-dependent execution is historical, from the completed Phase 2 Colab allocation. Publication checks hash the unchanged scientific sources, inspect syntax, validate evidence, and regenerate figures; they do not rerun inference. [SCIENTIFIC_SOURCE_MANIFEST.json](SCIENTIFIC_SOURCE_MANIFEST.json) records the preserved implementation bytes. The raw timing census is 92 records: 65 direct comparisons to fresh-original references, 25 resident requests linked to accepted first outputs, and two original workload-oracle runs. [Methods](docs/METHODS.md) defines that accounting and the exact comparison scope.
