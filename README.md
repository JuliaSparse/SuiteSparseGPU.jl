# SuiteSparse_GPU.jl (prototype)

Sparse Cholesky and sparse least squares for Julia, on NVIDIA GPUs. It runs the
SuiteSparse solvers behind `SparseArrays` (CHOLMOD and SPQR) with their CUDA acceleration,
from the CUDA build in `SuiteSparse_GPU_jll`. Requires Julia 1.13 or later, Linux, and an
NVIDIA driver supporting CUDA 13.

## Installation

SuiteSparse_GPU is not registered yet. Install it from GitHub:

```julia
using Pkg
Pkg.add(url = "https://github.com/JuliaSparse/SuiteSparse_GPU.jl")
```

or, in the Pkg REPL (press `]`):

```
pkg> add https://github.com/JuliaSparse/SuiteSparse_GPU.jl
```

This also installs `SuiteSparse_GPU_jll` and the CUDA runtime it needs; no CUDA toolkit
is required.

## Setup

```julia
using LinearAlgebra, SparseArrays, SuiteSparse_GPU

has_cuda()   # true if the GPU build is available on this platform

# Let each Cholesky factorization context reserve at most half of the GPU memory, so the
# QR solver has room too. Set this before the first GPU factorization.
ENV["CHOLMOD_GPU_MEM_FRACTION"] = "0.5"

# Test matrix: 7-point Laplacian on a k×k×k grid, shifted to be positive definite.
function laplace3d(k)
    D = spdiagm(-1 => -ones(k - 1), 0 => fill(2.0, k), 1 => -ones(k - 1))
    Ik = sparse(1.0I, k, k)
    return kron(kron(D, Ik), Ik) + kron(kron(Ik, D), Ik) + kron(kron(Ik, Ik), D) + 1e-2I
end
```

## Cholesky

```julia
A = laplace3d(60)          # 216,000 × 216,000
b = rand(size(A, 1))

F = gpu_cholesky(A)        # a regular CHOLMOD.Factor, like cholesky(A)
isgpu(F)                   # true
x = F \ b
norm(A * x - b) / norm(b)
logdet(F)

# New values with the same sparsity pattern: refactorize on the GPU.
gpu_cholesky!(F, A + 2I)
```

Compare with the CPU:

```julia
Fc = gpu_cholesky(A; gpu = false)
@time gpu_cholesky!(Fc, A; gpu = false)   # CPU
@time gpu_cholesky!(F, A)                 # GPU
```

`gpu_cholesky` takes a `SparseMatrixCSC`, or a `Symmetric` or `Hermitian` view of one,
real or complex. The gain grows with the size of the factor; small matrices gain little.

Refactorization (`gpu_cholesky!`) of `laplace3d(k)`, best of 3, AMD EPYC 9354 (32 threads)
vs one H100 PCIe (`bench/bench.jl`):

| grid  |         n |    nnz(L) | CPU s | GPU s | speedup |
|-------|----------:|----------:|------:|------:|--------:|
| 40³   |    64,000 |   15.5e6  |  0.32 |  0.29 |    1.1× |
| 60³   |   216,000 |   88.3e6  |  1.77 |  1.06 |    1.7× |
| 80³   |   512,000 |  301.2e6  |  5.74 |  3.20 |    1.8× |
| 100³  | 1,000,000 |  810.9e6  | 15.11 |  6.89 |    2.2× |
| 120³  | 1,728,000 |  1.77e9   | 31.77 | 11.63 |    2.7× |

### `cholesky` itself (Julia 1.14 and later)

On Julia 1.14 and later, loading SuiteSparse_GPU also switches `SparseArrays` to the GPU
build, so `cholesky` can use the GPU too. Load the package before the first sparse
factorization in the session.

```julia
F = with_gpu(() -> cholesky(A))   # this call only
isgpu(F)                          # true

use_gpu!(true)                    # every cholesky from here on
F = cholesky(A)
use_gpu!(false)
```

## Least squares (QR)

`qr_solve(A, B)` solves `A \ B` with a sparse QR factorization on the GPU. `qr(A)` does not
use the GPU.

```julia
n = 25^3
M = [laplace3d(25); sprandn(n ÷ 2, n, 5 / n)]   # 23,437 × 15,625
c = rand(size(M, 1))

x = qr_solve(M, c)                              # minimizes ‖M*x - c‖
norm(M' * (M * x - c)) / norm(c)                # ≈ 0 at the least squares solution

X = qr_solve(M, [c 2c])                         # several right-hand sides
```

Compare with the CPU:

```julia
@time qr_solve(M, c; gpu = false)   # CPU
@time qr_solve(M, c)                # GPU
```

The GPU is used for real matrices. By default `qr_solve` assumes `A` has full column rank;
for a rank deficient `A`, pass a tolerance, which runs on the CPU:

```julia
x = qr_solve(M, c; tol = 1e-10)
```

Least squares with `[laplace3d(k); sprandn(n/2, n, 5/n)]`, same machine
(`bench/bench_qr.jl`):

| grid |      n |  CPU s | GPU s | speedup |
|------|-------:|-------:|------:|--------:|
| 20³  |  8,000 |   2.80 |  0.82 |    3.4× |
| 25³  | 15,625 |  12.82 |  2.72 |    4.7× |
| 30³  | 27,000 |  36.63 |  8.21 |    4.5× |
| 35³  | 42,875 | 104.49 | 24.60 |    4.2× |

## GPU memory

The Cholesky solver keeps the GPU memory it reserves for the rest of the session. Without `CHOLMOD_GPU_MEM_FRACTION` (see Setup) it takes nearly all of it, and
`qr_solve` then warns and runs on the CPU. Restart Julia and set the fraction first to
use both.

## License

SuiteSparse_GPU.jl is MIT licensed (see [LICENSE.md](LICENSE.md)). The SuiteSparse
libraries it uses from `SuiteSparse_GPU_jll` include GPL-licensed code: CHOLMOD's GPU
module and SPQR.
