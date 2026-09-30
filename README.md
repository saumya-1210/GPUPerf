# GPU Performance Engineering Platform

A comprehensive CUDA kernel benchmarking, roofline modeling, Nsight Compute profiling, and LLM inference analysis platform — built and tested on T4/P100 GPUs.

![Python](https://img.shields.io/badge/Python-3.10+-3d9abf?style=flat-square)
![CUDA](https://img.shields.io/badge/CUDA-12.1-76b900?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-e8503a?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-f5c842?style=flat-square)

---

## Overview

I built this project to deeply understand GPU performance at the hardware level — not just running benchmarks, but understanding *why* a kernel performs the way it does and what the hardware is actually doing under the hood.

The architecture follows a systematic reasoning loop:

```
Hardware Limits → Roofline Model → Kernel Profiling → Optimization → Validation
```

This started as a CUDA learning exercise and evolved into a complete benchmarking and analysis platform covering everything from low-level GEMM optimization to LLM inference cost modeling.

---

## Key Results (T4 GPU)

| Measurement | Value | Interpretation |
|------------|-------|----------------|
| HBM Bandwidth (measured) | 280 GB/s | 87.5% of spec (320 GB/s) |
| Naive GEMM vs cuBLAS | 65 vs 44,100 GFLOPS | 0.1% efficiency — OI=0.25, pure BW-bound |
| WMMA Tensor Core vs cuBLAS | 33,000 vs 44,100 GFLOPS | 74.8% efficiency |
| K0 → K4 speedup | 508× | Through 4 targeted optimizations |
| Llama-7B decode OI (bs=1) | 0.4 FLOP/Byte | Memory-bound: compute is idle |
| Llama-7B decode throughput (bs=16) | ~890 tok/s | 32× better than bs=1 |
| Prefill scaling exponent | 1.94 | Confirms O(N²) attention |

---

## Modules

### 1. CUDA Kernel Engineering — `cuda_kernels/matmul_suite.cu`

Five progressively optimized GEMM kernels for N=4096 on T4, each targeting a specific bottleneck:

| Kernel | Technique | GFLOPS (T4) | % cuBLAS | Bottleneck Removed |
|--------|-----------|-------------|----------|-------------------|
| K0: Naive | Baseline | 65 | 0.1% | — |
| K1: Shared Memory Tiled | 32×32 SHMEM tile | 5,800 | 13.1% | Redundant HBM reads (32× reduction) |
| K2: Register Tiled | 4×4 output tile/thread | 6,900 | 15.6% | SHMEM→register bandwidth |
| K3: Vectorized | `float4` global loads | 7,000 | 15.9% | Instruction count for loads |
| K4: WMMA Tensor Core | FP16 WMMA API | 33,000 | 74.8% | FP32 → tensor core pipeline |
| cuBLAS (reference) | CUTLASS (vendor) | 44,100 | 100% | — |

> **Why K0 achieves only 0.1% of peak:** Naive GEMM arithmetic intensity is 0.25 FLOP/Byte (two HBM loads per FMA, no data reuse). T4 bandwidth ceiling at this OI = 0.25 × 280 = 70 GFLOPS. The kernel is purely bandwidth-bound before any compute happens.

```bash
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

```python
from roofline.roofline_model import HardwareRoof, build_demo_workloads, plot_roofline

roofs = [HardwareRoof.t4_measured(), HardwareRoof.a100_spec(), HardwareRoof.h100_spec()]
plot_roofline(roofs, build_demo_workloads(), 'roofline.png')
```

---

### 3. Nsight Compute Integration — `profiling/nsight_profiler.py`

Parses `ncu --csv` output and extracts the 12 metrics that matter most across 5 bottleneck categories:

| Metric | Category | What It Tells You |
|--------|----------|-------------------|
| `sm__throughput` | Compute | Are SMs busy? <50% → something else is limiting |
| `stall_long_scoreboard` | Memory | % cycles waiting for L2/HBM — the #1 memory-bound signal |
| `stall_math_throttle` | Compute | % cycles on FP pipeline — the "good" stall |
| `stall_barrier` | Sync | `__syncthreads()` overhead; >20% → try double-buffering |
| `pipe_tensor_cycles_active` | Tensor Cores | TC utilization; 0% on FP16 = missed opportunity |
| `l1_hit_rate / l2_hit_rate` | Cache | Tiling effectiveness |

The rule-based recommendation engine maps patterns to specific fixes:
- High `stall_long_scoreboard` → add tiling, increase cache reuse
- Low occupancy + high `registers_per_thread` → add `__launch_bounds__`
- High `stall_barrier` → double-buffer with async pipeline (Ampere+)
- Low `tensor_core_pct` on FP16 → switch to WMMA API

```python
# With real ncu output:
report = NsightReport.from_csv('ncu_raw.csv')

# Without GPU (demo mode with calibrated synthetic data):
report = NsightReport._demo_report()
visualize_nsight_report(report, 'nsight_analysis.png')
```

**Note:** Profiling with `ncu` requires a real CUDA driver (not the stub used in Colab). Kaggle's P100 tier and bare-metal instances work fine. Demo mode is explicitly labeled in all outputs.

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

Batching to 16 loads the same 14 GB of weights but processes 16 tokens simultaneously → ~32× throughput improvement.

**Supported models:** GPT-2, GPT-2 XL, Llama-2 7B/13B/70B, Llama-3 8B/70B, Mistral 7B, Mixtral 8×7B, GPT-4 (estimate)

**Supported GPUs:** T4, L4, A100-40/80, H100 SXM/PCIe, RTX 4090, RTX 5090, B200

**Outputs:** VRAM breakdown (weights + KV cache + activations), prefill/decode latency, tokens/sec, cost per 1M tokens, bottleneck classification

---

### 5. Report Generation — `report_gen/report_generator.py`

Runs all analyses and generates Markdown + HTML + JSON reports:

```bash
python main.py --module report
# → report_output/gpu_research_report.md
# → report_output/gpu_research_report.html
# → report_output/benchmark_results.json
```

---

## Getting Started

```bash
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

```bash
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
│   └── nsight_profiler.py       ← ncu CSV parser + visualization + recommendations
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

**Why does the naive MatMul only achieve 0.1% of peak FLOPS?**
> Arithmetic intensity of naive NxN GEMM = 0.25 FLOP/Byte (two HBM loads per FMA, no data reuse). T4 bandwidth ceiling at OI=0.25 is 0.25 × 280 = 70 GFLOPS. I measure 65 GFLOPS — 93% of the bandwidth ceiling, exactly what the roofline predicts. The kernel is not compute-limited at all.

**Why does tiling reduce memory traffic by 32×?**
> A 32×32 thread block computes 1,024 output elements. Without tiling: 2 × 32 × 1,024 = 65,536 HBM loads. With tiling: two 32×32 tiles (2,048 elements) loaded once, reused across 32 accumulation steps. Reuse factor = tile side length = 32.

**How do you know LLM decode is memory-bound?**
> At batch=1, each decode step loads ~14 GB of weights to compute ~5B FLOPs → OI ≈ 0.36 FLOP/Byte. T4's FP16 ridge point = 280 GB/s ÷ 65 TFLOPS ≈ 4.3 FLOP/Byte. Since 0.36 << 4.3, the GPU is idle on compute for >99% of the time. The memory bus is the bottleneck, not the tensor cores.

**What does `stall_long_scoreboard` mean in Nsight?**
> It measures the percentage of warp cycles stalled waiting for a long-latency memory operation — specifically an L2 or HBM access after a cache miss. When this metric exceeds 40%, the kernel is memory-latency-bound. The fix is to add tiling (increase cache reuse) or software prefetching to overlap memory latency with computation.

---

## Future Work

- [ ] Add `cp.async` path for sm80+ (Ampere async copy pipeline)
- [ ] Benchmark larger tile sizes (64×64, 128×128) to find occupancy/tile tradeoff
- [ ] Add CUTLASS comparison — understand the gap between K4 and cuBLAS at the instruction level
- [ ] Occupancy tuning via `__launch_bounds__` on K2/K3
- [ ] Run `ncu --set full` on real hardware and feed actual counter data into the nsight module

---

## License

MIT


