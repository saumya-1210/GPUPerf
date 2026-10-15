# Tesla T4 CUDA Kernel Profiling Analysis

## Hardware and tools

- GPU: NVIDIA Tesla T4
- GPU memory: 15,360 MiB
- Driver: 580.82.07
- CUDA: 13.0
- Nsight Compute: 2025.3.1.0
- Matrix size: N = 4096

## Kernel launch configurations

| Kernel | Grid size | Block size |
|---|---|---|
| K0 Naive | (256, 256, 1) | (16, 16, 1) |
| K1 Shared tiled | (128, 128, 1) | (32, 32, 1) |
| K2 Register tiled | (64, 64, 1) | (16, 16, 1) |
| K3 Vectorized | (128, 128, 1) | (8, 32, 1) |
| K4 WMMA Tensor Core | (256, 64, 1) | (32, 4, 1) |

## Nsight Compute metrics

| Metric | K0 | K1 | K2 | K3 | K4 |
|---|---:|---:|---:|---:|---:|
| SM throughput (%) | 61.292529 | 75.748728 | 50.703651 | 24.869838 | 17.626741 |
| DRAM throughput (%) | 18.029933 | 12.520668 | 20.736660 | 14.859752 | 38.614370 |
| Active warps (%) | 99.892079 | 99.980412 | 74.445194 | 99.693955 | 97.915591 |
| Issue active (%) | 17.512692 | 20.919982 | 41.473536 | 13.094922 | 5.496241 |
| Tensor pipe cycles active (%) | 0 | 0 | 0 | 0 | 5.996032 |
| Registers per thread | 49 | 42 | 72 | 62 | 64 |
| Shared memory per block (KB) | 0 | 8.192 | 4.096 | 8.192 | 0 |

## Metric definitions

- **SM throughput:** SM throughput relative to the metric's sustained-peak reference.
- **DRAM throughput:** device-memory throughput relative to the metric's sustained-peak reference.
- **Active warps:** active warp level relative to the metric's sustained-active reference.
- **Issue active:** instruction-issue activity relative to the metric's sustained-peak reference.
- **Tensor pipe cycles active:** Tensor pipe activity. K4's 5.996032% is not a percentage of theoretical Tensor Core FLOPS.
- **Registers per thread / shared memory per block:** kernel resource usage.

## Observations

- **K0 — Naive:** High active-warps percentage (99.89%) but lower issue-active value (17.51%). High warp activity alone does not establish efficient execution.
- **K1 — Shared tiled:** Highest SM throughput (75.75%) among these captured launches, with 8.192 KB shared memory per block.
- **K2 — Register tiled:** Highest issue-active value (41.47%), 72 registers per thread, and 74.45% active warps. This is consistent with a possible resource/occupancy trade-off, but these metrics alone do not prove the cause.
- **K3 — Vectorized:** SM throughput of 24.87% and issue-active of 13.09% in this capture. Vectorization alone did not make it the fastest kernel in the separate unprofiled benchmark.
- **K4 — WMMA:** Highest DRAM throughput in this table (38.61%) and non-zero Tensor pipe activity (6.00%). Tensor pipe activity is not the same as theoretical compute utilization.

These are observations from the captured counters, not definitive bottleneck diagnoses.

## Benchmark and validation context

- Unprofiled benchmark: 5 warm-up iterations, 20 timed iterations per kernel, and 3 runs.
- Use the separate benchmark results for latency and GFLOPS comparisons. Do not use profiler-instrumented timings to claim speedups.
- Independent CPU-reference validation passed for all five kernels at N = 64. This does not establish exact full-size FP32 equivalence.
- K4 uses FP16 inputs and FP32 accumulation, evaluated using its configured tolerance.

## Source

CUDA implementation: `cuda_kernels/matmul_suite.cu`
