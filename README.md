# GPU Performance Engineering Platform

A comprehensive CUDA kernel benchmarking, roofline modeling, Nsight Compute profiling, and LLM inference analysis platform — built and tested on T4/P100 GPUs.

[Python](https://camo.githubusercontent.com/1fb8baa116152369c9ee75b4a43430a9d5fd46e04b93130975b166dfb5ee049b/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f507974686f6e2d332e31302b2d3364396162663f7374796c653d666c61742d737175617265) [CUDA](https://camo.githubusercontent.com/857db0b28a58b54ca71c0a38464d670bf5a767a52b991f258fcf77445ee86673/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f435544412d31322e312d3736623930303f7374796c653d666c61742d737175617265) [PyTorch](https://camo.githubusercontent.com/210ea8a473f781c4765a94d647bdc41149bcb7c5a64bb68dff89785b13130a72/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5079546f7263682d322e332d6538353033613f7374796c653d666c61742d737175617265) [License](https://camo.githubusercontent.com/ed3583224422aed468aa3b052fdfa79576a2582b52134cf9f432b951936de765/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f4c6963656e73652d4d49542d6635633834323f7374796c653d666c61742d737175617265)

---

## Overview

I built this project to deeply understand GPU performance at the hardware level — not just running benchmarks, but understanding *why* a kernel performs the way it does and what the hardware is actually doing under the hood.

The architecture follows a systematic reasoning loop:

```
Hardware Limits → Roofline Model → Kernel Profiling → Optimization → Validation

```

This started as a CUDA learning exercise and evolved into a complete benchmarking and analysis platform covering everything from low-level GEMM optimization to LLM inference cost modeling.

---

## Key Results (NVIDIA Tesla T4)

The following GEMM results are from three unprofiled benchmark runs at N=4096. Each run used 5 warm-up iterations and 20 timed iterations per kernel. Timings were measured with CUDA events. Profiling-instrumented timings are excluded.

| Measurement | Mean result | Notes |
|---|---:|---|
| K0: Naive | 341.44 ms, 403.12 GFLOPS | Baseline |
| K1: Shared-memory tiled | 164.27 ms, 837.72 GFLOPS | 2.08× faster than K0 |
| K2: Register tiled | 67.33 ms, 2,044.65 GFLOPS | 5.07× faster than K0 |
| K3: Vectorized | 134.72 ms, 1,021.64 GFLOPS | 2.53× faster than K0 |
| K4: WMMA Tensor Core | 63.84 ms, 2,157.94 GFLOPS | 5.35× faster than K0 |
| cuBLAS reference | 32.73 ms, 4,208.57 GFLOPS | 10.43× faster than K0 |
| CPU-reference validation | 5/5 kernels passed at N=64 | Separate correctness test |

These are measurements from the recorded Tesla T4 runs; results may vary with GPU state, driver, and environment. K4 uses FP16 inputs with FP32 accumulation and its configured tolerance. The N=64 CPU-reference test does not establish exact full-size FP32 equivalence.

---

## Modules

### 1. CUDA Kernel Engineering — `cuda_kernels/matmul_suite.cu`

Five progressively optimized GEMM kernels for N=4096 on T4, each targeting a specific bottleneck:

| Kernel | Technique | Mean latency (ms) | Mean GFLOPS | Speedup vs K0 |
|---|---|---:|---:|---:|
| K0: Naive | Baseline | 341.44 | 403.12 | 1.00× |
| K1: Shared Memory Tiled | Shared-memory tiling | 164.27 | 837.72 | 2.08× |
| K2: Register Tiled | Register tiling | 67.33 | 2,044.65 | 5.07× |
| K3: Vectorized | Vectorized memory operations | 134.72 | 1,021.64 | 2.53× |
| K4: WMMA Tensor Core | FP16 inputs, FP32 accumulation | 63.84 | 2,157.94 | 5.35× |
| cuBLAS (reference) | Vendor library | 32.73 | 4,208.57 | 10.43× faster than K0 |

> **Measured baseline:** In the recorded N=4096 Tesla T4 benchmark, K0 averaged 341.44 ms and 403.12 GFLOPS. Nsight Compute counters help investigate execution behavior, but these measurements alone do not establish a definitive memory- or compute-bound diagnosis.

```
nvcc -O3 -arch=sm_75 -lcublas cuda_kernels/matmul_suite.cu -o matmul_bench
./matmul_bench

# Profile with Nsight Compute:
ncu --set full -o matmul_profile ./matmul_bench
```

---

### 2. Roofline Model — `roofline/roofline_model.py`

Establishes hardware ceilings from calibration kernels (not spec sheets), then places each kernel at its measured `(FLOPs/Byte, GFLOPS)` operating point.

What makes this different from naive approaches:

1. Hardware ceilings derived from actual calibration kernels
2. Per-kernel FLOPs + DRAM bytes collected via `ncu` metrics
3. Each kernel placed at `(FLOPs/Byte, GFLOPS)` on the roofline chart
4. Classification as compute/memory/mixed bound with ±20% mixed zone around ridge
5. Generates specific optimization advice based on distance from ceiling

Demo workload values are calibrated to be physically consistent with T4 hardware — every `achieved_gflops` satisfies `achieved ≤ min(OI × bandwidth, peak_compute)`.

```
from roofline.roofline_model import HardwareRoof, build_demo_workloads, plot_roofline

roofs = [HardwareRoof.t4_measured(), HardwareRoof.a100_spec(), HardwareRoof.h100_spec()]
plot_roofline(roofs, build_demo_workloads(), 'roofline.png')
```

---

### 3. Nsight Compute Integration — `profiling/nsight_profiler.py`

Parses `ncu --csv` output and extracts the 12 metrics that matter most across 5 bottleneck categories:

| **Metric**                  | **Category** | **What It Tells You**                                    |
| --------------------------- | ------------ | -------------------------------------------------------- |
| `sm__throughput`            | Compute      | Are SMs busy? <50% → something else is limiting          |
| `stall_long_scoreboard`     | Memory       | % cycles waiting for L2/HBM — the #1 memory-bound signal |
| `stall_math_throttle`       | Compute      | % cycles on FP pipeline — the "good" stall               |
| `stall_barrier`             | Sync         | `__syncthreads()` overhead; >20% → try double-buffering  |
| `pipe_tensor_cycles_active` | Tensor Cores | TC utilization; 0% on FP16 = missed opportunity          |
| `l1_hit_rate / l2_hit_rate` | Cache        | Tiling effectiveness                                     |

The rule-based recommendation engine maps patterns to specific fixes:

- High `stall_long_scoreboard` → add tiling, increase cache reuse
- Low occupancy + high `registers_per_thread` → add `__launch_bounds__`
- High `stall_barrier` → double-buffer with async pipeline (Ampere+)
- Low `tensor_core_pct` on FP16 → switch to WMMA API

```
# With real ncu output:
report = NsightReport.from_csv('ncu_raw.csv')

# Without GPU (demo mode with calibrated synthetic data):
report = NsightReport._demo_report()
visualize_nsight_report(report, 'nsight_analysis.png')
```

**Note:** This project includes a real Nsight Compute capture summary for the recorded Tesla T4 session in `profiling/tesla_t4_kernel_analysis.md`. Nsight Compute availability depends on the runtime's driver and permissions; demo mode remains available for environments where hardware profiling is unavailable. The profiling metrics are analyzed separately from unprofiled benchmark timings.

---

### 4. LLM Inference Analysis — `llm_profiler/inference_profiler.py`

Derives prefill/decode behavior from first principles. The core insight this module demonstrates:

**LLM decode at batch=1 is memory-bandwidth-limited on every GPU today.**

```
Bytes loaded per decode step  = model_weights + KV_cache  (~14 GB for Llama-7B FP16)
FLOPs per decode step         = 2 × num_layers × 4 × hidden_dim²  (~5B FLOPs)
Effective OI                  = 5e9 / 14e9 ≈ 0.36 FLOP/Byte
T4 ridge point (FP16)         = 65 TFLOPS / 280 GB/s ≈ 232 FLOP/Byte

0.36 << 232 → deeply memory-bound, compute is idle

```

Batching to 16 loads the same 14 GB of weights but processes 16 tokens simultaneously → \~32× throughput improvement.

**Supported models:** GPT-2, GPT-2 XL, Llama-2 7B/13B/70B, Llama-3 8B/70B, Mistral 7B, Mixtral 8×7B, GPT-4 (estimate)

**Supported GPUs:** T4, L4, A100-40/80, H100 SXM/PCIe, RTX 4090, RTX 5090, B200

**Outputs:** VRAM breakdown (weights + KV cache + activations), prefill/decode latency, tokens/sec, cost per 1M tokens, bottleneck classification

---

### 5. Report Generation — `report_gen/report_generator.py`

Runs all analyses and generates Markdown + HTML + JSON reports:

```
python main.py --module report
# → report_output/gpu_research_report.md
# → report_output/gpu_research_report.html
# → report_output/benchmark_results.json
```

---

## Getting Started

```
pip install -r requirements.txt

# Run specific modules:
python main.py --module roofline    # roofline chart
python main.py --module nsight      # Nsight analysis (demo data)
python main.py --module llm         # LLM profiler
python main.py --module kernels     # kernel benchmark table

# Full report:
python main.py

# Custom LLM profile:
python main.py --module llm --model llama_70b --gpu H100_SXM
```

### With a GPU

```
# Compile CUDA kernels (T4 = sm_75, A100 = sm_80, H100 = sm_90)
nvcc -O3 -arch=sm_75 -lcublas cuda_kernels/matmul_suite.cu -o matmul_bench
./matmul_bench

# Profile with Nsight Compute:
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,\
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed \
--csv --page raw ./matmul_bench > ncu_raw.csv

# Use real profile data:
python main.py --module nsight --nsight-csv ncu_raw.csv
```

---

## Project Structure

```
├── cuda_kernels/
│   └── matmul_suite.cu          ← K0–K4 GEMM kernels + cuBLAS benchmark
├── profiling/
│   ├── nsight_profiler.py       ← ncu CSV parser + visualization + recommendations
│   └── tesla_t4_kernel_analysis.md ← Captured Tesla T4 Nsight Compute metrics
├── roofline/
│   └── roofline_model.py        ← Roofline model with workload classification
├── llm_profiler/
│   └── inference_profiler.py    ← First-principles VRAM + latency + cost model
├── report_gen/
│   └── report_generator.py      ← Automated MD/HTML/JSON report generation
├── main.py                      ← Unified entry point
└── requirements.txt

```

---

## Known Limitations

- `cp.async` (software pipelining for Ampere+) is not yet implemented in K3
- K3 vectorizes global loads but computation remains scalar; a full register-tiled + vectorized kernel would score higher
- Benchmarks validated primarily on NVIDIA T4; sm_86/sm_90 behavior may differ
- Nsight Compute profiling (`ncu --csv`) is blocked on Colab T4 due to libcuda stub restrictions — the nsight module ships with calibrated demo data and a real-CSV path for environments where `ncu` works
- Hopper-specific features (TMA, WGMMA) are outside the current scope
- LLM latency estimates assume MFU=0.35 (reasonable for PyTorch + FlashAttention); real numbers vary ±20%

---

## Technical Deep Dives

**What did the measured kernel comparison show?**

> Across three unprofiled runs at N=4096, K0 averaged 341.44 ms (403.12 GFLOPS) and K4 averaged 63.84 ms (2,157.94 GFLOPS), a 5.35× speedup over K0. cuBLAS averaged 32.73 ms (4,208.57 GFLOPS). These values describe the recorded Tesla T4 environment and should not be treated as universal hardware limits.

**Why does tiling reduce memory traffic by 32×?**

> A 32×32 thread block computes 1,024 output elements. Without tiling: 2 × 32 × 1,024 = 65,536 HBM loads. With tiling: two 32×32 tiles (2,048 elements) loaded once, reused across 32 accumulation steps. Reuse factor = tile side length = 32.

**How do you know LLM decode is memory-bound?**

> At batch=1, each decode step loads \~14 GB of weights to compute \~5B FLOPs → OI ≈ 0.36 FLOP/Byte. T4's FP16 ridge point = 280 GB/s ÷ 65 TFLOPS ≈ 4.3 FLOP/Byte. Since 0.36 << 4.3, the GPU is idle on compute for >99% of the time. The memory bus is the bottleneck, not the tensor cores.

**What does `stall_long_scoreboard` mean in Nsight?**

> It measures the percentage of warp cycles stalled waiting for a long-latency memory operation — specifically an L2 or HBM access after a cache miss. When this metric exceeds 40%, the kernel is memory-latency-bound. The fix is to add tiling (increase cache reuse) or software prefetching to overlap memory latency with computation.

---

## Future Work

- &#x20;Add `cp.async` path for sm80+ (Ampere async copy pipeline)
- &#x20;Benchmark larger tile sizes (64×64, 128×128) to find occupancy/tile tradeoff
- &#x20;Add CUTLASS comparison — understand the gap between K4 and cuBLAS at the instruction level
- &#x20;Occupancy tuning via `__launch_bounds__` on K2/K3
- &#x20;Run `ncu --set full` on real hardware and feed actual counter data into the nsight module

---

## License

MIT

