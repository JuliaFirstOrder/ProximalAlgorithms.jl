using Test
using LinearAlgebra
using ProximalAlgorithms
using Random

@testset "CG" begin
    @testset "Real inputs" begin
        n = 100
        A = rand(n,n)
        A = A'A + I  # Make SPD
        b = rand(n)
        x0 = zeros(n)
        
        # Test basic CG
        cg = ProximalAlgorithms.CG(x0=x0, A=A, b=b)
        x, it = cg()
        @test norm(A*x - b) < 1e-6
        
        # Test with preconditioner
        P = Diagonal(diag(A))  # Jacobi preconditioner
        pcg = ProximalAlgorithms.CG(x0=x0, A=A, b=b, P=P)
        x, it = pcg()
        @test norm(A*x - b) < 1e-6
    end
    
    @testset "Complex inputs" begin
        n = 100
        A = rand(ComplexF64, n,n)
        A = A'A + I  # Make SPD
        b = rand(ComplexF64, n)
        x0 = zeros(ComplexF64, n)
        
        cg = ProximalAlgorithms.CG(x0=x0, A=A, b=b)
        x, it = cg()
        @test norm(A*x - b) < 1e-6
    end
    
    @testset "Custom operator" begin
        # Define simple operator that implements mul!
        struct DiagonalOperator{T}
            diag::Vector{T}
        end
        
        function LinearAlgebra.mul!(y, A::DiagonalOperator, x)
            y .= A.diag .* x
            return y
        end

        function Base.:*(A::DiagonalOperator, x)
            return A.diag .* x
        end
        
        n = 100
        d = rand(n) .+ 1  # Ensure positive diagonal
        A = DiagonalOperator(d)
        b = rand(n)
        x0 = zeros(n)
        
        cg = ProximalAlgorithms.CG(x0=x0, A=A, b=b)
        x, it = cg()
        @test norm(d .* x - b) < 1e-6
    end
    
    @testset "Ridge regression" begin
        n = 100
        A = rand(n, n)
        A = A'A + I  # Make SPD
        b = rand(n)
        x0 = zeros(n)
        λ = 0.1  # Regularization parameter

        # Test ridge regression with CG
        cg = ProximalAlgorithms.CG(x0=x0, A=A, b=b, λ=λ)
        x, it = cg()
        @test norm(A * x - b)^2 + λ * norm(x)^2 < norm(A * x0 - b)^2 + λ * norm(x0)^2

        # Test ridge regression with complex inputs
        A = rand(ComplexF64, n, n)
        A = A'A + I  # Make SPD
        b = rand(ComplexF64, n)
        x0 = zeros(ComplexF64, n)

        cg = ProximalAlgorithms.CG(x0=x0, A=A, b=b, λ=λ)
        x, it = cg()
        @test norm(A * x - b)^2 + λ * norm(x)^2 < norm(A * x0 - b)^2 + λ * norm(x0)^2
    end

    @testset "Per-element λ" begin
        # An array λ weights each unknown: CG solves (A + diag(λ)) x = b and CGNR
        # (AᴴA + diag(λ)) x = Aᴴb.
        n = 40
        A = rand(n, n)
        A = A'A + I
        b = rand(n)
        λ = rand(n)
        x, _ = ProximalAlgorithms.CG(x0 = zeros(n), A = A, b = b, λ = λ, maxit = 500)()
        @test x ≈ (A + Diagonal(λ)) \ b rtol = 1.0e-6
        B = rand(ComplexF64, 2n, n)
        c = rand(ComplexF64, 2n)
        x, _ = ProximalAlgorithms.CGNR(x0 = zeros(ComplexF64, n), A = B, b = c, λ = λ, maxit = 500)()
        @test x ≈ (B'B + Diagonal(λ)) \ (B'c) rtol = 1.0e-6
        @test ProximalAlgorithms.CG(x0 = zeros(n), A = A, b = b, λ = zeros(n), maxit = 500)()[1] ≈ A \ b rtol = 1.0e-6
    end
end
