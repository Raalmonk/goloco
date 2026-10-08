# Reusable GOLOCO inference

This backend takes compressed CERES features and produces GOLOCO's genome-wide prediction table. It keeps models in memory between requests and lets you choose Python or Rust for prediction.

For 18,141 targets and the three supplied experiment columns, the Rust backend with a loaded bundle took 1.35 seconds per resident request; optimized Python took 5.78 seconds. That is a 4.26× ratio of medians across five requests each. First output took 62.06 seconds with Rust and 56.62 seconds with Python-fast. Python-fast suits a one-off run; Rust pays off when the process stays alive.

[Results](docs/RESULTS.md) · [Methods](docs/METHODS.md) · [Figures](figures/README.md) · [Limitations](docs/LIMITATIONS.md) · [Attribution](docs/ATTRIBUTION.md)

## Inference API

Follow the [setup instructions](docs/REPRODUCE.md) for the Linux environment, author models, native library and bundle conversion. Then run this from the repository root with `PYTHONPATH=optimization`. Keep the instance open and call `engine.predict` again for each new input.

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

Choose `numpy` for the original estimator with indexed input preparation, `python_fast` for the original validator and a serial loop over scikit-learn's native trees, or `rust` for the native traversal. Rust requires a compatible library and has no automatic fallback. Omit `model_bundle` to export native parameters from source models as they enter the cache.

Set `bundle_mode="load"` explicitly for the configuration measured above. The constructor and CLI default to `mmap`; ordinary loading was slightly faster in this workload. Keep `chunk_size=1`. The experimental `ChunkedGoloco` class remains available, but [chunks of 64 and 256 targets were slower](experiments/chunk_comparison.py).

Every backend still needs the original source estimators for input validation. The cache holds models, metadata, identities and immutable native arrays. Each call reads the current input and computes new predictions. An instance accepts one operation at a time and rejects overlapping calls. [Data availability](docs/DATA_AVAILABILITY.md) lists the model downloads and files included here.

## Benchmarks

The [matched-residency comparison](figures/matched_residency.png) measured the existing Rust kernel at 0.93 seconds against a Python control at 6.37 seconds. Its 6.85× result reproduces the earlier kernel comparison; the packaged API above includes additional bookkeeping.

The [source manifest](SCIENTIFIC_SOURCE_MANIFEST.json) records the accepted implementation hashes. All 92 timing records are included: 65 direct checks against fresh original outputs, 25 resident checks linked to accepted first outputs, and two original workload-oracle runs. [Methods](docs/METHODS.md) describes the comparisons and historical model-dependent tests.

To rebuild the SVG, PDF and 300-dpi PNG figures and their tables, use the command in [figures/README.md](figures/README.md). This reads the committed records in a separate plotting environment. Publication checks cover source hashes, syntax, evidence and figure generation; inference ran in the completed Phase 2 experiment.
