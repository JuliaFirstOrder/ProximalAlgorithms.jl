using LinearAlgebra

function f_model(f_x, grad_f_x, res, L)
    return f_x - real(dot(grad_f_x, res)) + (L / 2) * norm(res)^2
end

function lower_bound_smoothness_constant(f, A, x, grad_f_Ax)
    R = real(eltype(x))
    xeps = x .+ 1
    grad_f_Axeps, _ = gradient(f, A * xeps)
    return norm(A' * (grad_f_Axeps - grad_f_Ax)) / R(sqrt(length(x)))
end

function lower_bound_smoothness_constant(f, A, x)
    Ax = A * x
    grad_f_Ax, _ = gradient(f, Ax)
    return lower_bound_smoothness_constant(f, A, x, grad_f_Ax)
end

_mul!(y, L, x) = mul!(y, L, x)
_mul!(y, ::Nothing, x) = return

_gradient!(y, f, x) = gradient!(y, f, x)
_gradient!(::Nothing, f, x) = f(x)

function backtrack_stepsize!(
    gamma::R, f, A, g, x, f_Ax::R, At_grad_f_Ax, y, z, g_z::R, res, Az, grad_f_Az=nothing;
    alpha = 1, minimum_gamma = 1e-7
) where R
    f_Az_upp = f_model(f_Ax, At_grad_f_Ax, res, alpha / gamma)
    _mul!(Az, A, z)
    f_Az = _gradient!(grad_f_Az, f, Az)
    tol = 10 * eps(R) * (1 + abs(f_Az))
    while f_Az > f_Az_upp + tol && gamma >= minimum_gamma
        gamma /= 2
        y .= x .- gamma .* At_grad_f_Ax
        g_z = prox!(z, g, y, gamma)
        res .= x .- z
        f_Az_upp = f_model(f_Ax, At_grad_f_Ax, res, alpha / gamma)
        _mul!(Az, A, z)
        f_Az = _gradient!(grad_f_Az, f, Az)
        tol = 10 * eps(R) * (1 + abs(f_Az))
    end
    if gamma < minimum_gamma
        @warn "stepsize `gamma` became too small ($(gamma))"
    end
    return gamma, g_z, f_Az, f_Az_upp
end

function backtrack_stepsize!(
    gamma, f, A, g, x; alpha = 1, minimum_gamma = 1e-7
)
    Ax = A * x
    grad_f_Ax, f_Ax = gradient(f, Ax)
    At_grad_f_Ax = A' * grad_f_Ax
    y = x - gamma .* At_grad_f_Ax
    z, g_z = prox(g, y, gamma)
    return backtrack_stepsize!(
        gamma, f, A, g, x, f_Ax, At_grad_f_Ax, y, z, g_z, x - z, Ax, grad_f_Ax;
        alpha = alpha, minimum_gamma = minimum_gamma
    )
end
