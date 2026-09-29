using Test, LinearAlgebra, SparseArrays, SuiteSparse_GPU

# Needs an NVIDIA GPU. Cap each CHOLMOD pool so the several Commons below (this package's,
# and on Julia 1.14 SparseArrays' for two tasks) leave device memory for SPQR.
ENV["CHOLMOD_GPU_MEM_FRACTION"] = "0.2"

# 7-point Laplacian on an n×n×n grid, shifted to be well conditioned.
function laplace3d(n)
    D = spdiagm(-1 => -ones(n - 1), 0 => fill(2.0, n), 1 => -ones(n - 1))
    I_ = sparse(1.0I, n, n)
    return kron(kron(D, I_), I_) + kron(kron(I_, D), I_) + kron(kron(I_, I_), D) + 1e-2I
end

@test has_cuda()

@testset "gpu_cholesky" begin
    A = laplace3d(40)
    b = rand(size(A, 1))
    xcpu = cholesky(A) \ b

    Fcpu = gpu_cholesky(A; gpu = false)
    @test !isgpu(Fcpu)
    @test Fcpu \ b ≈ xcpu

    F = gpu_cholesky(A)
    @test F isa SparseArrays.CHOLMOD.Factor{Float64,Int64}
    @test isgpu(F)
    @test gpu_stats().device_bytes > 0
    @test F \ b ≈ xcpu
    @test norm(A * (F \ b) - b) / norm(b) < 1e-10
    @test logdet(F) ≈ logdet(cholesky(A))

    # Refactorization reuses the GPU analysis.
    @test gpu_cholesky!(F, A + I) === F
    @test isgpu(F)
    @test (A + I) * (F \ b) ≈ b
    gpu_cholesky!(F, A; shift = 1.0)
    @test (A + I) * (F \ b) ≈ b

    # Other input types are converted to what the GPU path supports.
    F32 = gpu_cholesky(SparseMatrixCSC{Float32,Int32}(A))
    @test isgpu(F32)
    @test A * (F32 \ b) ≈ b rtol = 1e-5
    @test Symmetric(sparse(triu(A))) * (gpu_cholesky(Symmetric(sparse(triu(A)))) \ b) ≈ b

    E = sprand(size(A)..., 1e-5)
    Ac = A + im * 1e-3 * (E - E')
    Fc = gpu_cholesky(Hermitian(Ac))
    @test isgpu(Fc)
    @test Ac * (Fc \ b) ≈ b

    # A new task gets its own Common.
    @test fetch(Threads.@spawn isgpu(gpu_cholesky(A)))

    @test_throws ArgumentError gpu_cholesky(A + sparse(1:3, 4:6, 1.0, size(A)...))
    @test_throws PosDefException gpu_cholesky(A - 10I)
    @test !issuccess(gpu_cholesky(A - 10I; check = false))
end

@testset "qr_solve" begin
    # Overdetermined, so SPQR solves a least squares problem.
    M = [laplace3d(20); sprandn(4000, 8000, 1e-3)]
    c = M * ones(8000)
    B = [c rand(size(M, 1))]
    launches() = gpu_stats().qr_kernel_launches

    k = launches()
    xcpu = qr_solve(M, c; gpu = false)
    @test launches() == k
    @test xcpu ≈ ones(8000)
    @test xcpu ≈ qr(M) \ c

    xgpu = qr_solve(M, c)
    @test launches() > k
    @test gpu_stats().qr_flops > 0
    @test xgpu ≈ xcpu
    Xgpu = qr_solve(M, B)
    @test size(Xgpu) == (8000, 2)
    @test Xgpu ≈ qr_solve(M, B; gpu = false)
    @test norm(M' * (M * Xgpu[:, 2] - B[:, 2])) / norm(B[:, 2]) < 1e-8

    # Int32 indices use the Int32 Common.
    k32 = gpu_stats(Int32).qr_kernel_launches
    @test qr_solve(SparseMatrixCSC{Float64,Int32}(M), c) ≈ xcpu
    @test gpu_stats(Int32).qr_kernel_launches > k32

    # Rank detection and complex matrices stay on the CPU.
    k = launches()
    @test qr_solve(M, c; tol = -2) ≈ xcpu
    Mc = M + im * 1e-3 * M
    @test qr_solve(Mc, c) ≈ qr(Mc) \ c
    @test launches() == k

    @test_throws DimensionMismatch qr_solve(M, rand(3))
end

if VERSION >= v"1.14-"
    @testset "SparseArrays cholesky (Julia 1.14)" begin
        A = laplace3d(40)
        b = rand(size(A, 1))

        Fcpu = with_gpu(() -> cholesky(A); on = false)
        @test !isgpu(Fcpu)

        Fgpu = with_gpu(() -> cholesky(A))
        @test isgpu(Fgpu)
        @test Fgpu \ b ≈ Fcpu \ b

        with_gpu(() -> cholesky!(Fgpu, A + I))
        @test (A + I) * (Fgpu \ b) ≈ b

        # A fresh task picks up the setting from `use_gpu!`.
        use_gpu!(true)
        gpu = fetch(Threads.@spawn isgpu(cholesky(A)))
        use_gpu!(false)
        @test gpu

        # Unsupported cases fall back to the CPU silently.
        F32 = with_gpu(() -> cholesky(SparseMatrixCSC{Float64,Int32}(A)))
        @test !isgpu(F32)
        @test A * (F32 \ b) ≈ b
    end
end
