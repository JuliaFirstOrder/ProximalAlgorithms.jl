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

function backtrack_stepsize!(
    gamma, f, A, g, x, f_Ax::R, At_grad_f_Ax, y, z, g_z, res,
    alpha = 1, minimum_gamma = 1e-7
) where R
    while gamma >= minimum_gamma
        f_Az_upp = f_model(f_Ax, At_grad_f_Ax, res, alpha / gamma)
        Az = A * z
        grad_f_Az, f_Az = gradient(f, Az)
        tol = 10 * eps(R) * (1 + abs(f_Az))
        if f_Az <= f_Az_upp + tol
            return gamma, g_z, Az, f_Az, grad_f_Az, f_Az_upp
        end
        gamma /= 2
        y .= x .- gamma .* At_grad_f_Ax
        g_z = prox!(z, g, y, gamma)
        res .= x .- z
    end
    @warn "stepsize `gamma` became too small ($(gamma))"
    return gamma, g_z, Az, f_Az, grad_f_Az, f_Az_upp
end

function backtrack_stepsize!(
    gamma, f, A, g, x, alpha = 1, minimum_gamma = 1e-7
)
    Ax = A * x
    grad_f_Ax, f_Ax = gradient(f, Ax)
    At_grad_f_Ax = A' * grad_f_Ax
    y = x - gamma .* At_grad_f_Ax
    z, g_z = prox(g, y, gamma)
    return backtrack_stepsize!(
        gamma, f, A, g, x, f_Ax, At_grad_f_Ax, y, z, g_z, x - z,
        alpha, minimum_gamma
    )
end
