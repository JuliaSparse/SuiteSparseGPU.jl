# Run with CUDA_VISIBLE_DEVICES selecting the GPU.
using LinearAlgebra, SparseArrays, SuiteSparse_GPU, Printf

function laplace3d(n)
    D = spdiagm(-1 => -ones(n - 1), 0 => fill(2.0, n), 1 => -ones(n - 1))
    I_ = sparse(1.0I, n, n)
    return kron(kron(D, I_), I_) + kron(kron(I_, D), I_) + kron(kron(I_, I_), D) + 1e-2I
end

best(f) = minimum(@elapsed(f()) for _ in 1:3)

println("BLAS threads: ", BLAS.get_num_threads())
@printf("%-8s %10s %12s %10s %10s %8s\n", "grid", "n", "nnz(L)", "CPU s", "GPU s", "speedup")
for g in (40, 60, 80, 100, 120)
    A = laplace3d(g)
    F = gpu_cholesky(A; gpu = false)
    tc = best(() -> gpu_cholesky!(F, A; gpu = false))
    Fg = gpu_cholesky(A)
    tg = best(() -> gpu_cholesky!(Fg, A))
    @assert isgpu(Fg)
    @printf("%-8s %10d %12d %10.3f %10.3f %7.1fx\n", "$(g)^3", size(A, 1), nnz(F), tc, tg, tc / tg)
end

