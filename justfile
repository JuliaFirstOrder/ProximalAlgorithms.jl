julia:
    julia --project=.

instantiate:
    julia --project=. -e 'using Pkg; Pkg.instantiate()'

test:
    julia --project=. -e 'using Pkg; Pkg.test()'

format:
    julia --project=. -e 'using JuliaFormatter: format; format(".")'

docs:
    julia --project=./docs docs/make.jl
