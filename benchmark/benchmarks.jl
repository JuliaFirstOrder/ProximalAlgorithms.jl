using ProximalOperators
using ProximalAlgorithms
using BenchmarkTools
using LinearAlgebra
using SparseArrays
using Random

const SUITE = BenchmarkGroup()

for T in [Float64]
    k = "Lasso ($T)"
    SUITE[k] = BenchmarkGroup([k])
    A = T[
        1.0 -2.0 3.0 -4.0 5.0
        2.0 -1.0 0.0 -1.0 3.0
        -1.0 0.0 4.0 -3.0 2.0
        -1.0 -1.0 -1.0 1.0 3.0
    ]
    b = T[1.0, 2.0, 3.0, 4.0]
    m, n = size(A)
    R = real(T)
    lam = R(0.1) * norm(A' * b, Inf)

    SUITE[k]["ForwardBackward"] = @benchmarkable solver(x0, f=f, A=$A, g=g) setup=begin
        solver = ProximalAlgorithms.ForwardBackward(tol=1e-4)
        x0 = zeros($T, size($A, 2))
        f = Translate(SqrNormL2(), -$b)
        g = NormL1($lam)
    end

    SUITE[k]["FastForwardBackward"] = @benchmarkable solver(x0, f=f, A=$A, g=g) setup=begin
        solver = ProximalAlgorithms.FastForwardBackward(tol=1e-4)
        x0 = zeros($T, size($A, 2))
        f = Translate(SqrNormL2(), -$b)
        g = NormL1($lam)
    end

    SUITE[k]["ZeroFPR"] = @benchmarkable solver(x0, f=f, A=$A, g=g) setup=begin
        solver = ProximalAlgorithms.ZeroFPR(tol=1e-4)
        x0 = zeros($T, size($A, 2))
        f = Translate(SqrNormL2(), -$b)
        g = NormL1($lam)
    end

    SUITE[k]["PANOC"] = @benchmarkable solver(x0, f=f, A=$A, g=g) setup=begin
        solver = ProximalAlgorithms.PANOC(tol=1e-4)
        x0 = zeros($T, size($A, 2))
        f = Translate(SqrNormL2(), -$b)
        g = NormL1($lam)
    end

    SUITE[k]["DouglasRachford"] = @benchmarkable solver(x0, f=f, g=g) setup=begin
        solver = ProximalAlgorithms.DouglasRachford(tol=1e-4, gamma=$R(1))
        x0 = zeros($T, size($A, 2))
        f = LeastSquares($A, $b)
        g = NormL1($lam)
    end

    SUITE[k]["AFBA-1"] = @benchmarkable solver(x0, y0, f=f, g=g, beta_f=beta_f) setup=begin
        beta_f = opnorm($A)^2
        solver = ProximalAlgorithms.AFBA(theta=$R(1), mu=$R(1), tol=$R(1e-6))
        x0 = zeros($T, size($A, 2))
        y0 = zeros($T, size($A, 2))
        f = LeastSquares($A, $b)
        g = NormL1($lam)
    end

    SUITE[k]["AFBA-2"] = @benchmarkable solver(x0, y0, h=h, L=$A, g=g) setup=begin
        beta_f = opnorm($A)^2
        solver = ProximalAlgorithms.AFBA(theta=$R(1), mu=$R(1), tol=$R(1e-6))
        x0 = zeros($T, size($A, 2))
        y0 = zeros($T, size($A, 1))
        h = Translate(SqrNormL2(), -$b)
        g = NormL1($lam)
    end

    SUITE[k]["SFISTA"] = @benchmarkable solver(y0, f=f, Lf=Lf, h=h) setup=begin
        solver = ProximalAlgorithms.SFISTA(tol=$R(1e-3))
        y0 = zeros($T, size($A, 2))
        f = LeastSquares($A, $b)
        h = NormL1($lam)
        Lf = opnorm($A)^2
    end
end
