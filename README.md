# GPU-Accelerated Policy Inference and Motion Planning

Profiling and optimizing a robot manipulation policy inference loop 
on NVIDIA T4, from Python-level analysis to custom CUDA/Triton kernels.

## Results

| Experiment | Result |
|---|---|
| Roofline analysis | AI = 0.234 FLOP/byte — memory-bound |
| torch.compile | 11.8x latency reduction (2.13ms → 0.18ms) |
| Tiled CUDA matmul | 1.44x over naive, 5x gap to cuBLAS quantified |
| Triton flash-attention | 1.71x over PyTorch SDPA, numerically exact |

## Stack
PyTorch · CUDA C++ · Triton · RoboMimic · NVIDIA T4

## Structure
01_profiling/ → torch.profiler + roofline model  
02_torch_compile/ → compiler optimization benchmark  
03_cuda_kernels/ → naive and tiled matmul vs cuBLAS  
04_triton/ → flash-attention kernel implementation  
report/ → technical writeup
