# this script for comparing different solvers is inspired by
# - https://github.com/JuliaFirstOrder/StructuredOptimization.jl/blob/master/demos/LineSpectraEstimation.ipynb
# - https://github.com/JuliaSmoothOptimizers/RegularizedOptimization.jl/blob/paper/paper/examples/Benchmark.jl
# - https://juliafirstorder.github.io/ProximalAlgorithms.jl/stable/guide/custom_objectives/#Example:-counting-operations

using Random, SparseArrays, LinearAlgebra
using DSP, Statistics, FFTW
using Zygote
using DifferentiationInterface: AutoZygote
using ProximalCore, ProximalOperators, ProximalAlgorithms
using PrettyTables, LaTeXStrings

#--------------------------------------------------------------
# benchmark configuration
CFG = Dict(
    "seed" => 1234,
    "tol" => 1e-6,
    "max_iter" => 1_000_000,
    "verbose" => false,
    "solvers" => [:ForwardBackward, :FastForwardBackward, :ZeroFPR, :PANOC, :PANOCplus],
    "regularizers" => [:NormL1, :NormLinf, :NormL0, :IndBallL0],
    "print_table" => true,
)

#--------------------------------------------------------------
# counters
mutable struct Counting{T}
    f::T
    eval_count::Int
    grad_count::Int
    prox_count::Int
end

Counting(f::T) where {T} = Counting{T}(f, 0, 0, 0)

function (f::Counting)(x)
    f.eval_count += 1
    return f.f(x)
end

function ProximalAlgorithms.value_and_gradient(f::Counting, x)
    f.eval_count += 1
    f.grad_count += 1
    return ProximalAlgorithms.value_and_gradient(f.f, x)
end

function ProximalCore.prox!(y, f::Counting, x, gamma)
    f.prox_count += 1
    return ProximalCore.prox!(y, f.f, x, gamma)
end

#--------------------------------------------------------------
# problem generation
function make_loss_fun(cfg)
    Random.seed!(cfg["seed"]) # For reproducibility

    Fs = 16e3 #sampling frequency
    Nt = 2^7 #time samples
    y0 = zeros(Nt)
    SNR = 20
    S = 4 #super-resolution factor
    f_s = range(0, stop=Fs/2; length=div(S*Nt, 2)+1)     # super resolution frequency axis
    f = range(0, stop=Fs/2; length=div(Nt, 2)+1)       # frequency axis
    t = 0:(1/Fs):((Nt-1)/Fs)                                 # time axis
    N = 14 # number of sinusoids
    Nf = div(S*Nt, 4)+1
    p = randperm(Nf)[1:N]
    fk = f_s[p] # pick sinusoids frequencies 
    A = 0.1 .* randn(N) .+ 0.7   # amplitude
    for i in eachindex(fk)
        y0 .+= A[i] .* sin.(2*pi*fk[i] .* t) # mixture of sinusoids
    end
    y = y0 .+ 10^(-SNR/10)*sqrt(var(y0))*randn(length(y0)); # add noise

    x0 = randn(ComplexF64, S*Nt)
    f = ProximalAlgorithms.AutoDifferentiable(x -> 0.5*norm(ifft(x)[1:Nt]-complex(y), 2)^2, AutoZygote())
    return f, x0
end

function run_solver(f, g, x0, solversym, cfg)
    maxit = cfg["max_iter"]
    tol = cfg["tol"]
    verbose = cfg["verbose"]
    solverbuilder = ProximalAlgorithms.eval(solversym)
    solver = solverbuilder(maxit=maxit, tol=tol, verbose=verbose)

    fc = Counting(f)
    gc = Counting(g)

    runtime = @elapsed x, iters = solver(f=fc, g=gc, x0=x0)
    status = iters < maxit ? "first_order" : "max_iter"
    objx = f(x)+g(x)
    return (
        name=String(solversym),
        objective=objx,
        status=status,
        runtime=runtime,
        iters=iters,
        fevals=fc.eval_count,
        gradevals=fc.grad_count,
        proxevals=gc.prox_count
    )
end

function bench_regularizer(f, g, x0, cfg)
    results = NamedTuple[]
    for solversym in cfg["solvers"]
        push!(results, run_solver(f, g, x0, solversym, cfg))
    end

    restable = [
        (; name=m.name,
            status=string(m.status),
            runtime=round(m.runtime, digits=4),
            fevals=m.fevals,
            gradevals=m.gradevals,
            proxevals=m.proxevals,
            objx=round(m.objective, digits=4))
        for m in results
    ]

    return restable
end

# main
function main(; cfg=CFG, latex_out=true)
    f, x0 = make_loss_fun(cfg)

    for regsym in cfg["regularizers"]
        regstr = String(regsym)
        regbuilder = ProximalOperators.eval(regsym)
        if regbuilder == IndBallL0
            λ = 28 # 2*N in LineSpectraEstimation.ipynb
        else
            λ = 1
        end
        g = regbuilder(λ)

        restable = bench_regularizer(f, g, x0, cfg)

        if cfg["print_table"]
            println("\n=== Results with "*regstr*" ===")
            for row in restable
                println(row)
            end
        end

        # save as latex format
        if latex_out
            table_str = pretty_table(String,
                restable;
                backend=:latex,
                column_labels=["Method", "Status", L"$t$ ($s$)", L"$\#f$", L"$\#\nabla f$", L"$\#prox$", "Objective"],
                style=LatexTableStyle(column_label=String[]),
                table_format=latex_table_format__booktabs
            )

            open("table_"*regstr*".tex", "w") do io
                write(io, table_str)
            end
        end
    end

    return nothing
end