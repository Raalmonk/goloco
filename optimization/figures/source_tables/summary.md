# Generated evidence tables

All values come from committed per-sample records. Time is wall seconds. Quartiles use linear interpolation.

| Boundary | Workload | Arm | n | Median s | Min s | Max s | IQR s |
|---|---|---|---:|---:|---:|---:|---:|
| chunk_resident | original_three_experiments | chunk1 | 5 | 1.438062 | 1.424957 | 1.447280 | 0.009919 |
| chunk_resident | original_three_experiments | chunk256 | 5 | 5.053049 | 4.933187 | 5.072781 | 0.009625 |
| chunk_resident | original_three_experiments | chunk64 | 5 | 5.068635 | 5.038339 | 5.080720 | 0.026598 |
| matched_resident | original_three_experiments | B3 | 5 | 6.366502 | 5.289099 | 6.579939 | 0.619926 |
| matched_resident | original_three_experiments | C0 | 5 | 0.929296 | 0.924422 | 0.935394 | 0.003128 |
| original_request | original_three_experiments | A | 5 | 65.172071 | 64.979808 | 65.299667 | 0.121427 |
| pilot_oracle | 32_synthetic_performance_experiments | A | 1 | 3.870066 | 3.870066 | 3.870066 | 0.000000 |
| pilot_oracle | one_original_experiment | A | 1 | 3.863208 | 3.863208 | 3.863208 | 0.000000 |
| pilot_resident | 32_synthetic_performance_experiments | B3 | 5 | 0.413591 | 0.399454 | 0.417153 | 0.004781 |
| pilot_resident | 32_synthetic_performance_experiments | C0 | 5 | 0.154528 | 0.154275 | 0.155829 | 0.000398 |
| pilot_resident | one_original_experiment | B3 | 5 | 0.397903 | 0.394024 | 0.402439 | 0.001435 |
| pilot_resident | one_original_experiment | C0 | 5 | 0.074995 | 0.072815 | 0.076195 | 0.000849 |
| process_first_output | original_three_experiments | historical_c0 | 3 | 87.106980 | 86.778282 | 87.523698 | 0.372708 |
| process_first_output | original_three_experiments | packed_load | 3 | 62.064298 | 61.691387 | 62.112697 | 0.210655 |
| process_first_output | original_three_experiments | packed_mmap | 3 | 63.328703 | 62.942575 | 63.424257 | 0.240841 |
| process_first_output | original_three_experiments | source_python_fast | 3 | 56.618331 | 56.205505 | 57.192143 | 0.493319 |
| process_first_output | original_three_experiments | source_rust | 3 | 100.880155 | 99.663844 | 101.006150 | 0.671153 |
| standalone_resident | original_three_experiments | historical_c0 | 5 | 0.931193 | 0.926464 | 0.935571 | 0.003614 |
| standalone_resident | original_three_experiments | packed_load | 5 | 1.354753 | 1.340135 | 1.356393 | 0.002763 |
| standalone_resident | original_three_experiments | packed_mmap | 5 | 1.416301 | 1.403088 | 1.420206 | 0.005090 |
| standalone_resident | original_three_experiments | source_python_fast | 5 | 5.777371 | 5.760054 | 6.568567 | 0.573447 |
| standalone_resident | original_three_experiments | source_rust | 5 | 1.436388 | 1.430079 | 1.441735 | 0.005595 |

## Standalone serving memory

Maximum of five sampled resident-request RSS peaks per standalone process; GiB = bytes / 2^30. No error bars or distribution are inferred.

| Arm | Peak GiB |
|---|---:|
| source_python_fast | 7.428165 |
| source_rust | 10.802021 |
| packed_load | 10.844509 |
| packed_mmap | 10.861801 |

## Comparisons

- Matched B3 / C0 ratio of medians: 6.850887×.
- Packaged Python / packed-load ratio of resident medians: 4.264521×.
- One-time bundle conversion: 99.145200479 seconds.

Correctness: 65 direct fresh-original comparisons, 25 resident records linked to accepted first outputs, and 2 original workload-oracle records. Historical model-dependent tests were not rerun during publication.
