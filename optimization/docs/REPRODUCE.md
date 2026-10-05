# Reproduction

## Reproduce this publication without models

From the repository root, use the command in [figures/README.md](../figures/README.md) to create the separate plotting environment and regenerate figures/tables from committed records. Check accepted source hashes without importing any scientific package:

```bash
python3 optimization/scripts/check_sources.py
```

These are the fresh publication checks. No Colab runtime, inference, model conversion, or native build was performed during closeout. The commands below document independent reproduction of historical model-dependent work; they are not prerequisites for viewing or rebuilding the figures.

## Recreate the inference environment on Linux

The accepted code was executed with Python 3.8.20 and the complete [inference freeze](../environments/inference.lock.txt). From this fork's repository root, create an isolated environment; `uv` must already be installed:

```bash
uv python install 3.8.20
uv venv --python 3.8.20 .venv-inference
uv pip install --python .venv-inference/bin/python \
  -r optimization/environments/inference.lock.txt
export GOLOCO_PY="$PWD/.venv-inference/bin/python"
export GOLOCO_PROJECT="$PWD"
export GOLOCO_REPO="$GOLOCO_PROJECT/work/upstream"
export GOLOCO_OPT="$GOLOCO_PROJECT/optimization"
export PYTHONPATH="$GOLOCO_OPT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p work
git clone https://github.com/pritchardlabatpsu/goloco.git work/upstream
git -C work/upstream checkout --detach 470477c4be6ba0095016a2089b9bbf33d49a1ef8
```

The converter requires an unchanged tracked source checkout whose HEAD is exactly `470477c4be6ba0095016a2089b9bbf33d49a1ef8`. Keep that separate detached author checkout in `work/upstream`; the reusable package is imported from this fork's `optimization` directory. The new contribution commit is intentionally not passed as the converter's `--repo`. Do not silently substitute a new author-model archive or untested inference dependency version when comparing these results.

## Retrieve verified source models

Download the official archive outside version control, verify its exact bytes, and extract the L200 files. The helper refuses to overwrite an existing destination and does not unpickle:

```bash
curl --fail --location --output work/ceres-infer.zip \
  'https://zenodo.org/api/records/14251390/files/ceres-infer.zip/content'
"$GOLOCO_PY" optimization/scripts/prepare_models.py \
  --archive work/ceres-infer.zip --output work/models
export GOLOCO_MODELS="$GOLOCO_PROJECT/work/models"
export GOLOCO_MANIFEST="$GOLOCO_MODELS/author_model_manifest.json"
export GOLOCO_INPUT="$GOLOCO_REPO/data/PC9SKMELCOLO_L200_CERES_Features.csv"
export GOLOCO_TARGETS="$GOLOCO_OPT/evidence/canonical_targets.json"
```

The archive SHA-256 is `e02feb3118663982d74c9e8187dd47c5757686b6b3bcc59d090ba4fd6d8ad30e`; the other identities are in [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md). The output contains `L200_models/` and a per-file source manifest. Only trusted official model pickles belong in this workflow. The new preparation helper was checked as source during publication; it was not rerun against the 5.1 GB archive during closeout.

## Build the accepted native source

The historical compiler was Rust 1.99.0. Install that toolchain through your existing Rust setup if needed, then build on the intended Linux inference host:

```bash
bash optimization/scripts/build_rust.sh
export GOLOCO_LIBRARY="$GOLOCO_OPT/rust/target/release/libgoloco_forest.so"
```

The helper uses release mode, locked/offline Cargo, one build job, and `-C target-cpu=native`. There are no external Rust crates. The final source includes the unchanged per-target C0 kernel, a diagnostic entry point, and the experimental batch entry point. Core lifecycle measurements used the initial build; chunk measurements used the final build with unchanged core/profile functions. Build costs are separate from first-output timing, and machine-specific binaries are not committed.

## Single request and reusable bundle

Python-fast needs no Rust library or numeric bundle:

```bash
"$GOLOCO_PY" -m goloco_reusable predict \
  --repo "$GOLOCO_REPO" --models "$GOLOCO_MODELS" \
  --backend python_fast --input "$GOLOCO_INPUT" --targets "$GOLOCO_TARGETS" \
  --output work/python_prediction.csv --report work/python_prediction.json
```

To serve repeated requests using Rust, convert once, verify the completed bundle, then predict. Source estimators remain required. The bundle output directory must not already exist:

```bash
"$GOLOCO_PY" -m goloco_reusable convert \
  --repo "$GOLOCO_REPO" --models "$GOLOCO_MODELS" \
  --source-manifest "$GOLOCO_MANIFEST" --targets "$GOLOCO_TARGETS" \
  --output work/native_bundle
"$GOLOCO_PY" -m goloco_reusable verify \
  --bundle work/native_bundle --mode load --source-manifest "$GOLOCO_MANIFEST"
"$GOLOCO_PY" -m goloco_reusable predict \
  --repo "$GOLOCO_REPO" --models "$GOLOCO_MODELS" \
  --backend rust --bundle work/native_bundle --bundle-mode load --chunk-size 1 \
  --library "$GOLOCO_LIBRARY" --input "$GOLOCO_INPUT" --targets "$GOLOCO_TARGETS" \
  --output work/rust_prediction.csv --report work/rust_prediction.json
"$GOLOCO_PY" -m goloco_reusable validate \
  --repo "$GOLOCO_REPO" --models "$GOLOCO_MODELS" \
  --backend rust --bundle work/native_bundle --bundle-mode load --chunk-size 1 \
  --library "$GOLOCO_LIBRARY" --input "$GOLOCO_INPUT" --targets "$GOLOCO_TARGETS" \
  --report work/validation.json
```

`validate` performs fresh original inference, matrix checks, exact table and CSV comparison, and separate fixed-tolerance checks. It exits unsuccessfully if exact agreement fails. The shell invocations create separate processes; they do not demonstrate resident reuse. Use one persistent [Python API instance](../README.md#inference-api) for repeated requests. `prepare(targets)` performs loading explicitly and must be charged as startup. The CLI `--invalidate` option invalidates all entries; `--invalidate TARGET` selects one target. API methods support invalidation, snapshots, close, and reopen.

The preserved constructor and CLI default bundle mode is `mmap`; the examples select `load` explicitly because it was the faster packaged resident mode in this measurement. `numpy`, `python_fast`, and `rust` remain explicit choices. Optional chunking is a separate `goloco_reusable.chunk.ChunkedGoloco` API, not the accepted main CLI default.

## Historical model-dependent validation and benchmarks

The accepted drivers in `experiments/` and `reference/` are byte-identical to the historical sources and preserve their Colab/Linux guards. Several require `/content` and the optional chunk runner requires `/content/goloco_perf`; these are generic filesystem guards, not a saved runtime session. The drivers do not allocate or release a runtime. Use a new results directory and a newly chosen future measurement deadline; never replay a terminated allocation's deadline or controller.

Set `GOLOCO_DEADLINE` to the chosen Unix-epoch stop time and `GOLOCO_RSS_LIMIT` to the permitted process-and-descendant RSS bytes, no greater than 60% of currently available memory. Those values are caller inputs because setup time and available RAM belong to the new execution, not the saved evidence. The historical cache cap was 16 GiB. Commands below are Bash arrays:

```bash
: "${GOLOCO_DEADLINE:?Set a fresh future measurement deadline in epoch seconds}"
: "${GOLOCO_RSS_LIMIT:?Set the aggregate process RSS limit in bytes}"
GOLOCO_RESULTS="$GOLOCO_PROJECT/work/reproduction_results"
mkdir -p "$GOLOCO_RESULTS"
goloco_common=(
  --repo "$GOLOCO_REPO" --model-root "$GOLOCO_MODELS"
  --input "$GOLOCO_INPUT" --targets "$GOLOCO_TARGETS" --library "$GOLOCO_LIBRARY"
  --deadline-epoch "$GOLOCO_DEADLINE" --process-limit-bytes "$GOLOCO_RSS_LIMIT"
)
export GOLOCO_TEST_REPO="$GOLOCO_REPO"
export GOLOCO_TEST_MODEL_DIR="$GOLOCO_MODELS/L200_models"
export GOLOCO_TEST_MANIFEST="$GOLOCO_MANIFEST"
"$GOLOCO_PY" -m unittest discover -s "$GOLOCO_OPT/tests" -p test_bundle.py -v
"$GOLOCO_PY" "$GOLOCO_OPT/experiments/regression.py" "${goloco_common[@]}" \
  --bundle work/native_bundle --out "$GOLOCO_RESULTS/backend_regression" --count 16
"$GOLOCO_PY" "$GOLOCO_OPT/experiments/run_phase2.py" original "${goloco_common[@]}" \
  --reps 5 --out "$GOLOCO_RESULTS/original_full"
"$GOLOCO_PY" "$GOLOCO_OPT/experiments/run_phase2.py" paired "${goloco_common[@]}" \
  --reps 5 --oracle "$GOLOCO_RESULTS/original_full" --out "$GOLOCO_RESULTS/historical_pair"
"$GOLOCO_PY" "$GOLOCO_OPT/experiments/run_phase2.py" lifecycle "${goloco_common[@]}" \
  --bundle work/native_bundle --oracle "$GOLOCO_RESULTS/original_full" \
  --startup-reps 3 --warm-reps 5 --out "$GOLOCO_RESULTS/lifecycle"
```

The original output is a post-timing reference, never a substitute for a candidate's inference. Main lifecycle variants are `historical_c0`, `source_python_fast`, `source_rust`, `packed_load`, and `packed_mmap`. Five subsequent resident samples are taken only in trial 0 of each variant. Do not compare an imported resident clock to the parent launch-to-first-output boundary as if they were equivalent.

The preserved negative chunk comparison and input-selection audit have these interfaces:

```bash
"$GOLOCO_PY" "$GOLOCO_OPT/experiments/chunk_comparison.py" "${goloco_common[@]}" \
  --bundle work/native_bundle --bundle-mode mmap --oracle "$GOLOCO_RESULTS/original_full" \
  --out "$GOLOCO_RESULTS/chunk_comparison" --chunks 1,64,256 --reps 5
"$GOLOCO_PY" "$GOLOCO_OPT/experiments/workload_expansion.py" "${goloco_common[@]}" \
  --out "$GOLOCO_RESULTS/full_matrix_audit" --matrix-audit-only
"$GOLOCO_PY" "$GOLOCO_OPT/reference/edge_validation_native.py" \
  --library "$GOLOCO_LIBRARY" --tier-results "$GOLOCO_RESULTS/original_full" \
  --out "$GOLOCO_RESULTS/native_edges" --models 16 --deadline-seconds 170
```

The workload-expansion driver also retains the historical 1,024-target one-real/32-synthetic pilot and its measured-budget rule. Model-dependent commands are supplied for independent reproduction; the completed publication adds no new timing samples or test passes. Figure regeneration uses only the saved public records.
