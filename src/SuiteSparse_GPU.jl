"""
    SuiteSparse_GPU

GPU-accelerated sparse Cholesky (CHOLMOD) and sparse QR (SPQR) for `SparseArrays`
(JuliaSparse/SparseArrays.jl#443), using the CUDA build of SuiteSparse in
`SuiteSparse_GPU_jll`.

CHOLMOD offloads the dense BLAS/LAPACK work on large supernodes to CUDA, and SPQR runs the
whole factorization on the GPU, when `Common->useGPU` is set. There is no separate API: the
usual `cholmod_l_analyze`, `cholmod_l_factorize` and `SuiteSparseQR` calls use the GPU.

The JLL names its libraries `libcholmod_cuda` and `libspqr_cuda`, so they load next to the
CPU-only ones `SparseArrays` uses. [`gpu_cholesky`](@ref) and [`qr_solve`](@ref) call
them directly with this package's own task-local Commons, and work on every supported
Julia version. Both CHOLMODs allocate through the one `libsuitesparseconfig`, so
`gpu_cholesky` returns an ordinary `CHOLMOD.Factor`.

On Julia 1.14 and later, `SparseArrays` can also load its libraries from another directory
(`LibSuiteSparse.set_libdir!`). Loading this package then points it at the JLL, so
`cholesky` itself can use the GPU through [`with_gpu`](@ref) and [`use_gpu!`](@ref).
"""
module SuiteSparse_GPU

using LinearAlgebra: LinearAlgebra, Hermitian, Symmetric, PosDefException, issuccess
using SparseArrays: SparseArrays, SparseMatrixCSC, CHOLMOD
using SparseArrays.LibSuiteSparse: LibSuiteSparse, cholmod_common, cholmod_sparse,
    cholmod_dense, cholmod_factor, CHOLMOD_QUERY_HAS_CUDA, CHOLMOD_SUPERNODAL,
    SPQR_ORDERING_DEFAULT, SPQR_NO_TOL
import SuiteSparse_GPU_jll

export has_cuda, gpu_cholesky, gpu_cholesky!, qr_solve, isgpu, gpu_stats
@static if VERSION >= v"1.14-"
    export use_gpu!, with_gpu
end

# The JLL defines its library products only on platforms it was built for.
const libcholmod_cuda = isdefined(SuiteSparse_GPU_jll, :libcholmod) ?
    SuiteSparse_GPU_jll.libcholmod : "libcholmod_cuda"
const libspqr_cuda = isdefined(SuiteSparse_GPU_jll, :libspqr) ?
    SuiteSparse_GPU_jll.libspqr : "libspqr_cuda"

function check_available()
    SuiteSparse_GPU_jll.is_available() || throw(ArgumentError(
        "SuiteSparse_GPU_jll has no build for this platform ($(Sys.MACHINE))"))
    LibSuiteSparse.init_suitesparse()
    return nothing
end

"""
    has_cuda() -> Bool

Whether the CUDA build of CHOLMOD and SPQR from `SuiteSparse_GPU_jll` is available on this
platform.
"""
function has_cuda()
    SuiteSparse_GPU_jll.is_available() || return false
    LibSuiteSparse.init_suitesparse()
    return @ccall libcholmod_cuda.cholmod_l_query(CHOLMOD_QUERY_HAS_CUDA::Cint)::Bool
end

## Commons

# Without these, CHOLMOD reserves nearly all free device memory for each Common's pool at
# its first GPU analysis. CHOLMOD itself reads them only when it resolves `useGPU` from the
# `CHOLMOD_USE_GPU` environment variable.
function set_memory_limits!(c)
    b = get(ENV, "CHOLMOD_GPU_MEM_BYTES", nothing)
    b === nothing || (c[].maxGpuMemBytes = max(0, something(tryparse(Int, strip(b)), 0)))
    f = get(ENV, "CHOLMOD_GPU_MEM_FRACTION", nothing)
    f === nothing ||
        (c[].maxGpuMemFraction = clamp(something(tryparse(Float64, strip(f)), 0.0), 0, 1))
    return c
end

function error_handler(status::Cint, file::Cstring, line::Cint, message::Cstring)::Cvoid
    status < 0 && throw(CHOLMOD.CHOLMODException(unsafe_string(message)))
    return nothing
end

finish!(c::Ref{cholmod_common}, ::Type{Int64}) =
    @ccall libcholmod_cuda.cholmod_l_finish(c::Ptr{cholmod_common})::Cint
finish!(c::Ref{cholmod_common}, ::Type{Int32}) =
    @ccall libcholmod_cuda.cholmod_finish(c::Ptr{cholmod_common})::Cint

function newcommon(::Type{Ti}) where Ti
    c = Ref(cholmod_common())
    ok = Ti === Int64 ?
        @ccall(libcholmod_cuda.cholmod_l_start(c::Ptr{cholmod_common})::Cint) :
        @ccall(libcholmod_cuda.cholmod_start(c::Ptr{cholmod_common})::Cint)
    ok == 1 || error("cholmod_start failed")
    finalizer(c -> finish!(c, Ti), c)
    c[].print = 0
    c[].error_handler = @cfunction(error_handler, Cvoid, (Cint, Cstring, Cint, Cstring))
    return c
end

# Whether SparseArrays loads its CHOLMOD and SPQR from this JLL (Julia 1.14 and later).
const SHARES_SPARSEARRAYS_LIBS = Ref(false)

# The Commons for the CUDA libraries, one per task and index type. When SparseArrays loads
# the same libraries these are its own, so each task has one GPU memory pool; otherwise
# they are this package's.
function gpucommon(::Type{Ti}) where Ti<:Union{Int32,Int64}
    check_available()
    SHARES_SPARSEARRAYS_LIBS[] && return CHOLMOD.getcommon(Ti)
    key = Ti === Int64 ? :suitesparse_gpu_common_l : :suitesparse_gpu_common
    return get!(() -> newcommon(Ti), task_local_storage(), key)::Ref{cholmod_common}
end

# Run `f(c)` with the Common set up for one call, restoring the settings afterwards since
# the Common may be the one `cholesky` uses. The GPU is used only by supernodal
# factorizations.
function with_common(f, ::Type{Ti}, gpu::Bool) where Ti
    c = gpucommon(Ti)
    old = (c[].useGPU, c[].supernodal, c[].final_ll, c[].gpuMemorySize)
    try
        c[].useGPU = gpu
        c[].supernodal = CHOLMOD_SUPERNODAL
        c[].final_ll = 1
        gpu && set_memory_limits!(c)
        return f(c)
    finally
        c[].useGPU, c[].supernodal, c[].final_ll, c[].gpuMemorySize = old
    end
end

"""
    gpu_stats(Ti = Int64)

Return the GPU state of the current task's Common for index type `Ti`, used by
[`gpu_cholesky`](@ref) and [`qr_solve`](@ref) (and, on Julia 1.14 and later, by
`cholesky`): the device and pinned host memory CHOLMOD reserved for it, and the kernel
launches and flops SPQR ran on the GPU, summed over all `qr_solve` calls. CHOLMOD reserves
its pools at the first GPU analysis and keeps them until the task's Common is freed; they
exist only for `Int64`.
"""
function gpu_stats(::Type{Ti} = Int64) where Ti<:Union{Int32,Int64}
    c = gpucommon(Ti)[]
    return (; device_bytes = Int(c.dev_mempool_size),
              pinned_host_bytes = Int(c.host_pinned_mempool_size),
              qr_kernel_launches = Int(c.gpuNumKernelLaunches), qr_flops = Int(c.gpuFlops))
end

## Cholesky

const HermOrSymCSC{T} = Union{SparseMatrixCSC{T}, Symmetric{T,<:SparseMatrixCSC{T}},
                              Hermitian{T,<:SparseMatrixCSC{T}}}

cholmod_eltype(A) = eltype(A) <: Complex ? ComplexF64 : Float64

"""
    gpu_cholesky(A; shift = 0.0, check = true, gpu = true) -> CHOLMOD.Factor

Compute the supernodal Cholesky factorization of the sparse Hermitian positive definite
matrix `A`, like `cholesky(A)`, with CHOLMOD's CUDA build. `A` is a `SparseMatrixCSC` or a
`Symmetric`/`Hermitian` view of one, and is converted to `Float64` or `ComplexF64` values
and `Int64` indices, the types CHOLMOD's GPU path supports. The dense work on large
supernodes runs on the GPU; small matrices run almost entirely on the CPU.

The result is an ordinary `CHOLMOD.Factor`, supporting `\\`, `ldiv!`, `logdet` and the rest.
Refactorize it on the GPU with [`gpu_cholesky!`](@ref); `cholesky!` works too, but runs
on the CPU. With `gpu = false` the same library runs on the CPU, for comparison.
"""
function gpu_cholesky(A::HermOrSymCSC; shift::Real = 0.0, check::Bool = true,
                      gpu::Bool = true)
    S = CHOLMOD.Sparse{cholmod_eltype(A),Int64}(A)
    unsafe_load(pointer(S)).stype == 0 && throw(ArgumentError(
        "matrix is not Hermitian; wrap it in `Hermitian` or `Symmetric` to use one triangle"))
    L = with_common(Int64, gpu) do c
        @ccall libcholmod_cuda.cholmod_l_analyze(S::Ptr{cholmod_sparse},
            c::Ptr{cholmod_common})::Ptr{cholmod_factor}
    end
    F = CHOLMOD.Factor{cholmod_eltype(A),Int64}(L)
    return factorize!(F, S, shift, check, gpu)
end

"""
    gpu_cholesky!(F::CHOLMOD.Factor, A; shift = 0.0, check = true, gpu = true) -> F

Refactorize `F` for `A`, which has the same sparsity pattern as the matrix `F` was computed
from, reusing its symbolic analysis. This runs on the GPU for a factorization computed by
[`gpu_cholesky`](@ref).
"""
function gpu_cholesky!(F::CHOLMOD.Factor{Tv,Int64}, A::HermOrSymCSC; shift::Real = 0.0,
                       check::Bool = true, gpu::Bool = true) where Tv
    return factorize!(F, CHOLMOD.Sparse{Tv,Int64}(A), shift, check, gpu)
end

function factorize!(F::CHOLMOD.Factor, S::CHOLMOD.Sparse, shift, check, gpu)
    beta = Float64[shift, 0]
    with_common(Int64, gpu) do c
        @ccall libcholmod_cuda.cholmod_l_factorize_p(S::Ptr{cholmod_sparse},
            beta::Ptr{Cdouble}, C_NULL::Ptr{Int64}, 0::Csize_t, F::Ptr{cholmod_factor},
            c::Ptr{cholmod_common})::Cint
    end
    check && !issuccess(F) &&
        throw(PosDefException(Int(unsafe_load(pointer(F)).minor) + 1))
    return F
end

"""
    isgpu(F::CHOLMOD.Factor) -> Bool

Whether the symbolic analysis of `F` set it up for GPU factorization, so that numeric
(re)factorizations of `F` with GPU use on run on the GPU.
"""
function isgpu(F::CHOLMOD.Factor{<:Any,Ti}) where Ti
    Ti === Int64 || return false
    return unsafe_load(pointer(F)).useGPU != 0
end

## QR

# Free device memory for SPQR, which splits the factorization into stages that fit in it.
# CHOLMOD measures it only through an Int64 Common with `useGPU == 1`.
function gpu_free_memory()
    c = gpucommon(Int64)
    old = c[].useGPU
    total, avail = Ref{Csize_t}(0), Ref{Csize_t}(0)
    try
        c[].useGPU = 1
        @ccall libcholmod_cuda.cholmod_l_gpu_memorysize(total::Ptr{Csize_t},
            avail::Ptr{Csize_t}, c::Ptr{cholmod_common})::Cint
    finally
        c[].useGPU = old
    end
    return Int(avail[])
end

"""
    qr_solve(A::SparseMatrixCSC, B::AbstractVecOrMat; tol = -1, ordering, gpu = true)

Solve ``\\min\\|AX - B\\|`` with a sparse QR factorization from SPQR's CUDA build, computing
``Q'B`` along with `R` instead of storing `Q`, which SPQR's GPU path requires. For ``m < n``
the result is a basic solution, not the minimum norm one. `ordering` is one of the
`SparseArrays.SPQR.ORDERING_*` constants.

The GPU is used for real `A` with `Int32` or `Int64` indices and `tol < 0`, which turns
rank detection off: a rank deficient `A` then gives an inaccurate or infinite `X`. With
`tol ≥ 0`, or `tol ≤ -2` for SPQR's default, columns of 2-norm at most `tol` are treated
as zero and the factorization runs on the CPU, like `qr(A) \\ B`. With `gpu = false` it
always runs on the CPU.

SPQR allocates device memory for each call, which must not be held by CHOLMOD's pools; set
`CHOLMOD_GPU_MEM_FRACTION` or `CHOLMOD_GPU_MEM_BYTES` before the first GPU Cholesky
factorization to cap them. With less than 64 MiB free, `qr_solve` warns and uses the CPU.
"""
function qr_solve(A::SparseMatrixCSC{<:Any,Ti}, B::AbstractVecOrMat;
                  tol::Real = SPQR_NO_TOL, ordering::Integer = SPQR_ORDERING_DEFAULT,
                  gpu::Bool = true) where Ti
    size(A, 1) == size(B, 1) || throw(DimensionMismatch(
        "A has $(size(A, 1)) rows but B has $(size(B, 1))"))
    Tv = promote_type(float(eltype(A)), float(eltype(B)), Float64)
    Tj = Ti === Int32 ? Int32 : Int64
    S = CHOLMOD.Sparse(SparseMatrixCSC{Tv,Tj}(A), 0)
    D = CHOLMOD.Dense(convert(Matrix{Tv}, reshape(B, size(B, 1), :)))
    gpu = gpu && Tv === Float64 && tol < 0
    mem = gpu ? gpu_free_memory() : 0
    if gpu && mem <= 1
        @warn "Less than 64 MiB of GPU memory is free, so SPQR runs on the CPU. " *
            "Cap CHOLMOD's pools with CHOLMOD_GPU_MEM_FRACTION; see `qr_solve`." maxlog = 1
        gpu = false
    end
    X = with_common(Tj, gpu) do c
        c[].gpuMemorySize = mem
        CHOLMOD.Dense(@ccall libspqr_cuda.SuiteSparseQR_C_backslash(ordering::Cint,
            tol::Cdouble, S::Ptr{cholmod_sparse}, D::Ptr{cholmod_dense},
            c::Ptr{cholmod_common})::Ptr{cholmod_dense})
    end
    return B isa AbstractVector ? vec(Matrix(X)) : Matrix(X)
end

## SparseArrays' own `cholesky` (Julia 1.14 and later)

@static if VERSION >= v"1.14-"

# `LibSuiteSparse.set_libdir!` needs a directory holding the whole set of SuiteSparse
# libraries under SuiteSparse_jll's file names. Link SuiteSparse_jll's, with CHOLMOD and
# SPQR replaced by the JLL's CUDA builds. SparseArrays and this package then use the same
# CUDA CHOLMOD. (The CPU-only one is loaded too, by SuiteSparse_GPU_jll's dependency on
# SuiteSparse_jll, but nothing calls it.)
function point_sparsearrays_at_jll()
    SuiteSparse_GPU_jll.is_available() || return
    Base.generating_output() && return
    haskey(ENV, "JULIA_SUITESPARSE_LIBDIR") && return   # the user's choice wins
    src = LibSuiteSparse.libdir()
    dir = mktempdir(; prefix = "suitesparse_gpu_")
    pat = r"^lib(suitesparseconfig|amd|camd|colamd|ccolamd|cholmod|spqr|umfpack)\."
    for f in readdir(src)
        occursin(pat, f) || continue
        target = startswith(f, "libcholmod.") ? SuiteSparse_GPU_jll.libcholmod_path :
                 startswith(f, "libspqr.") ? SuiteSparse_GPU_jll.libspqr_path :
                 joinpath(src, f)
        symlink(target, joinpath(dir, f))
    end
    try
        LibSuiteSparse.set_libdir!(dir)
        SHARES_SPARSEARRAYS_LIBS[] = true
    catch err
        # SuiteSparse was used before this package was loaded.
        err isa ArgumentError || rethrow()
        @debug "SparseArrays keeps its CPU-only SuiteSparse" exception = err
    end
    return
end

"""
    SuiteSparse_GPU.set_libdir!(dir)

Load `SparseArrays`' SuiteSparse libraries from `dir`, which must hold a full SuiteSparse
build with CUDA under the same file names as `SuiteSparse_jll`, instead of the ones from
`SuiteSparse_GPU_jll` that loading this package selects. It must run before the first
sparse factorization; see `SparseArrays.LibSuiteSparse.set_libdir!`. Julia 1.14 and later.
"""
set_libdir!(dir::AbstractString) = LibSuiteSparse.set_libdir!(dir)

# Whether the libcholmod `SparseArrays` loaded was built with CUDA.
function sparsearrays_has_cuda()
    LibSuiteSparse.init_suitesparse()
    return LibSuiteSparse.cholmod_l_query(CHOLMOD_QUERY_HAS_CUDA)
end

function check_sparsearrays_cuda()
    sparsearrays_has_cuda() || throw(ArgumentError("SparseArrays loaded SuiteSparse " *
        "without CUDA from $(LibSuiteSparse.libdir()), because SuiteSparse was used " *
        "before `using SuiteSparse_GPU`; load it first, or use `gpu_cholesky`"))
    return nothing
end

# The Common `cholesky` uses for Int64 indices on the current task.
common() = CHOLMOD.getcommon(Int64)

# CHOLMOD reads this once per Common, at the first supernodal analysis done with it.
const USE_GPU_ENV = "CHOLMOD_USE_GPU"

"""
    use_gpu!(on::Bool = true)

Enable or disable GPU factorization for `SparseArrays`' `cholesky`. This sets
`CHOLMOD_USE_GPU`, which CHOLMOD reads for every task that has not factorized a supernodal
matrix yet, and sets `useGPU` directly on the current task's Common. Tasks that already
factorized with the other setting keep it; use [`with_gpu`](@ref) inside them. Julia 1.14
and later.
"""
function use_gpu!(on::Bool = true)
    on && check_sparsearrays_cuda()
    ENV[USE_GPU_ENV] = on ? "1" : "0"
    c = common()
    c[].useGPU = on
    on && set_memory_limits!(c)
    return on
end

"""
    with_gpu(f; on::Bool = true, supernodal::Bool = true)

Run `f()` with GPU factorization for `SparseArrays`' `cholesky` set to `on` for the current
task, restoring the previous setting afterwards. With `supernodal = true`, CHOLMOD is also
made to use the supernodal method, which it otherwise chooses only above a fill
threshold; the GPU is used only by supernodal factorizations. Julia 1.14 and later.
"""
function with_gpu(f; on::Bool = true, supernodal::Bool = true)
    on && check_sparsearrays_cuda()
    c = common()
    old_gpu, old_super = c[].useGPU, c[].supernodal
    try
        c[].useGPU = on
        on && set_memory_limits!(c)
        supernodal && (c[].supernodal = CHOLMOD_SUPERNODAL)
        return f()
    finally
        c[].useGPU = old_gpu
        c[].supernodal = old_super
    end
end

end # @static if

function __init__()
    @static if VERSION >= v"1.14-"
        point_sparsearrays_at_jll()
    end
end

end # module
