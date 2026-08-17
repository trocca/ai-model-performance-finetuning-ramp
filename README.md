# AI Model Performance & Fine-Tuning Ramp

A hands-on readiness track for benchmarking, profiling, optimizing, and
fine-tuning AI models—from training efficiency and inference performance to
reproducible evaluation and deployment.

The goal is not merely to call high-level APIs. By the end of this track, you
should be able to explain where model time and memory go, prove which
optimization helped, implement a performance-critical operation below PyTorch,
and defend the quality/cost trade-offs of a fine-tuned model.

## Target outcome

Build an evidence-backed portfolio showing that you can:

- establish reproducible training and inference baselines;
- interpret CPU, GPU, memory, and kernel profiler traces;
- improve throughput, latency, utilization, and memory without hiding quality
  regressions;
- select full fine-tuning, parameter-efficient fine-tuning, or quantized
  fine-tuning for a given constraint;
- write and validate a PyTorch C++/CUDA extension;
- optimize model execution with mixed precision, compilation, fusion,
  quantization, batching, and caching;
- evaluate a tuned model against its base model and a meaningful baseline;
- package the final model with a benchmark report and reproducible serving path.

## Language and framework choices

| Layer | Choice | Why it is here |
|---|---|---|
| Model development | **Python + PyTorch** | Best path for model experimentation, autograd, distributed training, profiling, compilation, and the current fine-tuning ecosystem. |
| Systems integration | **C++17/20** | Exposes tensor storage, dispatch, CPU kernels, memory ownership, extension boundaries, and low-overhead benchmark harnesses. |
| GPU optimization | **CUDA C++** | Gives direct control over kernels, memory traffic, occupancy, synchronization, streams, and Tensor Core execution. |
| Optional kernel DSL | **Triton** | Useful for rapid custom-kernel experiments before or alongside handwritten CUDA. It complements rather than replaces CUDA in this track. |

PyTorch is the primary framework. C++ and CUDA are used where crossing below the
framework boundary produces useful understanding or measurable performance.

## Performance vocabulary

Every benchmark must name the metric and workload. “Faster” is not a result.

| Metric | Meaning |
|---|---|
| Latency | Time for one request, batch, step, or generated token. Report percentiles for serving. |
| Throughput | Samples, tokens, sequences, or requests completed per second. |
| Time to first token (TTFT) | Delay before the first generated token is available. |
| Inter-token latency (ITL) | Time between generated tokens after the first one. |
| Training step time | Forward + loss + backward + optimizer work for one update. |
| Model FLOPs utilization (MFU) | Achieved model arithmetic relative to theoretical accelerator throughput under stated assumptions. |
| Memory footprint | Parameters, gradients, optimizer state, activations, temporary workspaces, and serving caches. |
| Quality | Task metric, loss, perplexity, human preference, or another workload-appropriate measure. |
| Cost efficiency | Useful work or quality per GPU-hour, watt-hour, or currency unit. |

## Rules for trustworthy measurements

1. Record hardware, driver, CUDA, compiler, PyTorch, precision, model revision,
   input shapes, batch size, sequence lengths, and random seed.
2. Separate cold start, warm-up, steady state, and compilation time.
3. Synchronize asynchronous GPU work around host-side timers.
4. Use enough warm-up and measured iterations; report median and tail latency,
   not only the best run.
5. Keep input generation and data loading out of kernel timing unless the test is
   explicitly end to end.
6. Compare outputs against a trusted reference with defined tolerances.
7. Track model quality with performance. An optimization that silently harms the
   target metric is a trade-off, not a free speedup.
8. Store raw benchmark results in machine-readable form and generate tables or
   plots from those files.

## The theory map

### Model computation

- tensor shapes, strides, layouts, broadcasting, and dtype conversion;
- forward computation graphs, reverse-mode autodiff, saved activations, and
  gradient accumulation;
- dense layers, convolutions, normalization, activation functions, attention,
  embedding lookup, and loss functions;
- Transformer prefill versus autoregressive decode;
- arithmetic intensity and the roofline model.

### GPU execution

- grids, blocks, warps, threads, and SIMT execution;
- global, shared, constant, and register memory;
- coalescing, bank conflicts, occupancy, divergence, and launch overhead;
- reductions, tiling, fusion, asynchronous copies, streams, and events;
- Tensor Cores, mixed precision, accumulation precision, and numerical stability.

### Training performance

- batch size, micro-batches, gradient accumulation, and global batch size;
- data-pipeline overlap, pinned memory, worker processes, and host bottlenecks;
- FP32, TF32, FP16, BF16, FP8, and loss scaling where applicable;
- optimizer state, activation checkpointing, gradient clipping, and memory
  fragmentation;
- data parallelism, sharding, communication, and compute/communication overlap.

### Fine-tuning

- domain adaptation, supervised fine-tuning, instruction tuning, and continued
  pretraining;
- catastrophic forgetting, overfitting, leakage, and train/eval contamination;
- full fine-tuning versus adapters, LoRA, QLoRA, prefix/prompt methods, and
  selective layer freezing;
- rank, alpha, adapter targets, learning rate, schedule, batch construction, and
  context length;
- evaluation design, ablations, uncertainty, and base-model comparison.

### Inference performance

- batching, dynamic batching, padding waste, and continuous batching;
- KV-cache sizing, reuse, paging, and memory bandwidth;
- quantization formats, calibration, group size, scales, and quality loss;
- operator fusion, graph compilation, kernel selection, CUDA Graphs, and memory
  planning;
- throughput/latency trade-offs and service-level objectives.

## Toolchain

### Core

- Python 3.11 or newer
- PyTorch with CUDA support
- CMake and Ninja
- a C++17/20 compiler compatible with the installed CUDA toolkit
- NVIDIA CUDA Toolkit and `nvcc`
- `pytest`, NumPy, pandas, Matplotlib, and Jupyter

### Profiling and diagnosis

- `torch.profiler` for framework operators, shapes, stacks, and memory;
- PyTorch memory snapshots and allocator statistics;
- NVIDIA Nsight Systems for the CPU/GPU timeline and gaps;
- NVIDIA Nsight Compute for kernel-level metrics;
- `nvidia-smi` and DCGM for device state, clocks, power, and utilization;
- Linux `perf` for CPU-side work where available;
- sanitizers and `compute-sanitizer` for correctness and race detection.

### Fine-tuning and evaluation

- Hugging Face Transformers, Datasets, Tokenizers, Accelerate, and PEFT;
- bitsandbytes where supported for low-bit optimizer and QLoRA experiments;
- a versioned experiment configuration format such as YAML;
- TensorBoard, MLflow, or Weights & Biases for experiment tracking;
- task-specific evaluation plus latency, throughput, memory, and cost reports.

### Serving and optimization

- PyTorch eager mode and `torch.compile` as the first comparison;
- ONNX Runtime or TensorRT where the model/export path supports them;
- vLLM for LLM serving and continuous-batching/KV-cache experiments;
- Triton Inference Server when a multi-model production serving exercise is useful.

## Repository shape

```text
.
├── README.md
├── environment/
│   ├── requirements.txt
│   └── system-report.sh
├── common/
│   ├── benchmark.py
│   ├── correctness.py
│   └── reporting.py
├── labs/
│   ├── 01-baseline-and-measurement/
│   ├── 02-pytorch-profiler/
│   ├── 03-data-pipeline/
│   ├── 04-mixed-precision-and-compile/
│   ├── 05-cpp-extension/
│   ├── 06-cuda-kernel/
│   ├── 07-full-finetuning/
│   ├── 08-lora-and-qlora/
│   ├── 09-evaluation-and-ablations/
│   ├── 10-inference-optimization/
│   ├── 11-serving-benchmark/
│   └── 12-capstone/
├── results/
│   ├── raw/
│   ├── figures/
│   └── reports/
└── notes/
```

Each lab should contain a short theory note, runnable baseline, optimized version,
correctness tests, raw results, and a conclusion stating what changed and why.

## Twelve-week readiness plan

### Week 1 — Baselines and measurement discipline

**Theory:** latency, throughput, warm-up, synchronization, variance, percentiles,
confidence, and experimental controls.

**Lab:** benchmark a small CNN and Transformer block in eager PyTorch. Record CPU
and GPU timing correctly, then demonstrate how missing synchronization produces a
false result.

**Gate:** one command reproduces a JSON result file containing environment,
configuration, samples, median, p95, throughput, and peak memory.

### Week 2 — PyTorch execution and memory

**Theory:** autograd graph, saved tensors, dispatcher, operators, allocator,
parameters, activations, gradients, and optimizer state.

**Lab:** profile one training step with `torch.profiler`; produce an operator table,
timeline, memory ledger, and explanation of the three largest costs.

**Gate:** reconcile measured memory with a hand estimate and explain the gap.

### Week 3 — Data-pipeline performance

**Theory:** storage, decoding, augmentation, workers, prefetching, pinned memory,
non-blocking copies, and overlap.

**Lab:** intentionally starve the GPU, then tune loader workers, prefetching,
pinning, and augmentation placement. Confirm the improvement in an Nsight Systems
timeline.

**Gate:** improve end-to-end throughput and show that the model computation itself
did not change.

### Week 4 — Precision, compilation, and fusion

**Theory:** TF32/FP16/BF16, accumulation, numerical error, autocast, graph breaks,
fusion, CUDA Graphs, and compile overhead.

**Lab:** compare eager FP32, mixed precision, and `torch.compile` on fixed workloads.
Report cold and steady-state results separately and test numerical agreement.

**Gate:** identify at least one graph break or fusion opportunity and prove its
impact with a trace.

### Week 5 — PyTorch C++ extension

**Theory:** ATen tensors, dispatcher registration, contiguous versus strided
layouts, ownership, CPU parallelism, build ABI, and autograd integration.

**Lab:** implement a fused CPU operation in C++, expose it to Python, and compare it
against a pure PyTorch reference for correctness and performance.

**Gate:** randomized tests cover shapes, dtypes, non-contiguous inputs, and failure
cases; benchmark crossover points are documented.

### Week 6 — CUDA extension and kernel optimization

**Theory:** indexing, coalescing, occupancy, reductions, shared memory, launch
configuration, fusion, and roofline analysis.

**Lab:** add a CUDA implementation of the Week 5 operation. Build a naive kernel,
profile it, optimize the dominant bottleneck, and integrate backward if applicable.

**Gate:** match the PyTorch reference within declared tolerances, pass
`compute-sanitizer`, and explain the speedup using profiler evidence rather than
only wall-clock timing.

### Week 7 — Full fine-tuning baseline

**Theory:** dataset splits, leakage, tokenization, loss masking, schedules,
regularization, checkpointing, and evaluation.

**Lab:** fully fine-tune a small model on a well-defined task. Record the untouched
base-model metric, training curves, validation metric, GPU memory, and GPU-hours.

**Gate:** reproduce the run from a versioned config and compare against a
non-neural or zero-/few-shot baseline where appropriate.

### Week 8 — LoRA and QLoRA

**Theory:** low-rank updates, adapter target modules, rank/alpha, trainable parameter
count, quantized base weights, and memory-quality trade-offs.

**Lab:** run controlled full-tuning, LoRA, and QLoRA experiments on the same data
split and evaluation suite.

**Gate:** publish a table covering quality, trainable parameters, peak memory,
training throughput, wall time, checkpoint size, and inference implications.

### Week 9 — Evaluation, ablation, and failure analysis

**Theory:** task metrics, calibration, bootstrap uncertainty, slice analysis,
contamination, regression tests, and human evaluation limits.

**Lab:** ablate LoRA rank, target modules, learning rate, and training-data volume.
Create failure slices and compare base versus tuned outputs blindly where possible.

**Gate:** make a defensible recommendation with uncertainty and known failure
modes—not merely the best single score.

### Week 10 — Inference optimization

**Theory:** prefill/decode, KV cache, batching, quantization, graph compilation,
memory bandwidth, and operator/kernel selection.

**Lab:** optimize inference for the tuned model using batching, compilation, and at
least one quantization path. Measure quality before and after optimization.

**Gate:** report TTFT, ITL, tokens/s, peak memory, and quality under identical input
distributions.

### Week 11 — Serving and load testing

**Theory:** online versus offline inference, queues, continuous batching, admission
control, concurrency, p50/p95/p99 latency, saturation, and backpressure.

**Lab:** serve the model with a simple PyTorch baseline and an optimized serving
runtime. Load-test concurrency and sequence-length distributions rather than one
fixed prompt.

**Gate:** identify the saturation point and propose an operating region that meets
a stated service-level objective.

### Week 12 — Capstone: tune, optimize, and defend

Choose a model and task with a real evaluation signal. Deliver:

1. an immutable base-model baseline;
2. a documented dataset and split strategy;
3. at least two fine-tuning approaches;
4. profiler-led training improvements;
5. an optimized inference path;
6. a load-tested serving configuration;
7. quality, performance, memory, and cost comparisons;
8. one C++/CUDA contribution with correctness tests;
9. a final report explaining rejected approaches and remaining risks.

**Final gate:** another engineer can clone the repository, reproduce the core
results, inspect raw measurements, and understand why the selected configuration
is the best trade-off for the stated goal.

## Required lab report template

Every lab report should answer:

```text
Question:
Hypothesis:
Hardware/software environment:
Model, data, shapes, precision, and batch configuration:
Baseline:
Change made:
Correctness validation:
Measurement method:
Raw result location:
Result with variance/percentiles:
Profiler evidence:
Quality impact:
Interpretation:
Limitations:
Next experiment:
```

## Portfolio artifacts

- reproducible benchmark harness with JSON/CSV output;
- PyTorch profiler and Nsight traces with written interpretations;
- memory accounting worksheet for training and inference;
- tested C++/CUDA PyTorch extension;
- full fine-tuning versus LoRA versus QLoRA comparison;
- evaluation suite with regression tests and failure slices;
- quantization/compilation inference report;
- serving load-test report;
- capstone model card and performance report.

## Readiness checklist

- [ ] I can distinguish latency, throughput, utilization, and quality metrics.
- [ ] I can time asynchronous CUDA work correctly.
- [ ] I can explain a PyTorch profiler trace from Python call to CUDA kernel.
- [ ] I can account for parameters, activations, gradients, and optimizer state.
- [ ] I can determine whether a workload is compute-, memory-, launch-, or
      input-bound.
- [ ] I can validate an optimized operation numerically against a reference.
- [ ] I can build and debug a PyTorch C++/CUDA extension.
- [ ] I can choose between full fine-tuning, LoRA, and QLoRA from constraints.
- [ ] I can design an evaluation that detects quality regressions and leakage.
- [ ] I can explain prefill, decode, KV caching, batching, and quantization.
- [ ] I can report p50/p95/p99 serving behavior under realistic load.
- [ ] I can defend a model configuration using reproducible evidence.

## Definition of done

This ramp is complete when the capstone demonstrates all four layers together:

```text
model quality
    + training efficiency
    + inference/serving performance
    + C++/CUDA systems evidence
```

The final claim should be narrow, measured, reproducible, and honest about the
hardware, data, assumptions, and remaining limitations.
