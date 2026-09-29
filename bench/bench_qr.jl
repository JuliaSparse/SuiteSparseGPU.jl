# Run like bench.jl. Kept separate because the pool CHOLMOD reserves in bench.jl would
# leave no device memory for SPQR.
using LinearAlgebra, SparseArrays, SuiteSparse_GPU, Printf

function laplace3d(n)
    D = spdiagm(-1 => -ones(n - 1), 0 => fill(2.0, n), 1 => -ones(n - 1))
    I_ = sparse(1.0I, n, n)
    return kron(kron(D, I_), I_) + kron(kron(I_, D), I_) + kron(kron(I_, I_), D) + 1e-2I
end

best(f) = minimum(@elapsed(f()) for _ in 1:3)

# Least squares with the Laplacian stacked on n/2 sparse random rows.
println("BLAS threads: ", BLAS.get_num_threads())
@printf("%-8s %10s %10s %10s %8s\n", "grid", "n", "CPU s", "GPU s", "speedup")
for g in (20, 25, 30, 35)
    A = laplace3d(g)
    n = size(A, 1)
    M = [A; sprandn(n ÷ 2, n, 5 / n)]
    b = rand(size(M, 1))
    tc = best(() -> qr_solve(M, b; gpu = false))
    k = gpu_stats().qr_kernel_launches
    tg = best(() -> qr_solve(M, b))
    @assert gpu_stats().qr_kernel_launches > k
    @printf("%-8s %10d %10.3f %10.3f %7.1fx\n", "$(g)^3", n, tc, tg, tc / tg)
end
