using Documenter, DocumenterCitations, ProximalAlgorithms
using Literate

bib = CitationBibliography(joinpath(@__DIR__, "references.bib"))

src_path = joinpath(@__DIR__, "src/")

Literate.markdown(joinpath(src_path, "getting_started.jl"), src_path, documenter=true)
Literate.markdown(joinpath(src_path, "custom_objectives.jl"), src_path, documenter=true)

makedocs(
    bib,
    modules=[ProximalAlgorithms],
    sitename="ProximalAlgorithms.jl",
    pages=[
        "Home" => "index.md",
        "User guide" => [
            "getting_started.md",
            "implemented_algorithms.md",
            "custom_objectives.md",
            "custom_algorithms.md",
        ],
        "Bibliography" => "bibliography.md",
    ],
)

deploydocs(
    repo="github.com/JuliaFirstOrder/ProximalAlgorithms.jl.git",
)
