# Test scope

`test_bundle.py` is byte-identical to the accepted Phase 2 test source. Its 18 historical tests passed in the isolated Colab inference environment. Tests include numeric bundle validation and ownership, corruption/truncation, format/ABI errors, deterministic conversion from two author models, and fresh-process reuse. The real-source test requires `GOLOCO_TEST_REPO`, `GOLOCO_TEST_MODEL_DIR`, and `GOLOCO_TEST_MANIFEST`.

The additional accepted regression sources are:

- `../experiments/regression.py`: backend contract and lifecycle/state checks.
- `../reference/edge_validation_native.py`: native boundaries and threshold probes.
- `../reference/supplemental_validation.py`: selection, cache, invalid-input, and identity checks.
- `../experiments/chunk_comparison.py`: optional batch contracts, ownership and cache-pressure checks, and the retained negative benchmark.
- `../experiments/workload_expansion.py`: full selected-matrix audit and separately labeled performance pilots.

These depend on the pinned Linux inference environment, trusted author models, and (for native paths) the compiled library. Their historical passes are not fresh publication test runs. Publication checks use `scripts/check_sources.py` and the evidence/figure workflow without importing the inference package. [REPRODUCE.md](../docs/REPRODUCE.md) lists the separate model-dependent commands.
