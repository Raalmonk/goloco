# Limitations

Rust is useful when a process handles repeated requests. Python-fast reached the first output sooner, and starting a new process for each request does not recover Rust's extra startup cost in the measured scenario. An existing native bundle still requires integrity checks and source-model loading; creating it adds a one-time conversion cost.

Memory use is substantial: standalone serving peaks were about 7.43 GiB for Python-fast and 10.80 to 10.86 GiB for Rust. Both load and mmap keep the original estimators. The cache budget charges the full numeric bundle capacity and is separate from measured RSS.

The backend still unpickles trusted source models for validation. Those models must match the verified bundle. A validator that works without model objects was neither implemented nor tested. Compatibility is tied to the recorded legacy Python stack and Linux native build. `-C target-cpu=native` produces a machine-specific binary; rebuild it on the intended host. Binaries, environments, model archives, and native bundles are not included in the fork.

Each backend instance accepts one operation at a time and rejects overlapping prediction or lifecycle calls. Parallel serving, production concurrency, GPU prediction, and service deployment were not tested.

The timings come from one CPU allocation, with five resident samples per main arm and three fresh-process trials per variant. OS caches were uncontrolled. These are descriptive measurements for the tested workloads; the sample sizes do not support p95 or statistical-significance claims. Chunks of 64 and 256 were slower than single-target calls despite reducing FFI calls and preserving exact outputs. The diagnostics do not explain the entire slowdown.

Correctness coverage comprises 65 direct checks, 25 resident checks against accepted first outputs, and two original oracle runs. Regression cases add coverage, but equivalence has not been tested for every possible input. Fresh-original comparisons were exact; the saved author CSV matched within tolerance. The expanded workload tests cover only 1,024 targets with one real column or 32 deterministic synthetic columns. Full-target expansions were skipped under the runtime-budget rule. These tests add no biological measurements or accuracy claims.

See [Methods](METHODS.md) for timing boundaries and correctness checks, and [Results](RESULTS.md) for measurements.
