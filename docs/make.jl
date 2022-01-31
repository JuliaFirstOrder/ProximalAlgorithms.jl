using Documenter, DocumenterCitations, ProximalAlgorithms
using Literate

bib = CitationBibliography(joinpath(@__DIR__, "references.bib"))

src_path = joinpath(@__DIR__, "src/")

literate_directories = joinpath.(
    src_path,
    [
        "guide",
        "examples",
    ]
)

for directory in literate_directories
    jl_files = filter(p -> endswith(p, ".jl"), readdir(directory; join=true))
    for src in jl_files
        Literate.markdown(src, directory, documenter=true)
    end
end

makedocs(
    bib,
    modules=[ProximalAlgorithms],
    sitename="ProximalAlgorithms.jl",
    pages=[
        "Home" => "index.md",
        "User guide" => [
            joinpath("guide", "getting_started.md"),
            joinpath("guide", "implemented_algorithms.md"),
            joinpath("guide", "custom_objectives.md"),
            joinpath("guide", "custom_algorithms.md"),
        ],
        "Examples" => [
            joinpath("examples", "sparse_linear_regression.md"),
        ],
        "Bibliography" => "bibliography.md",
    ],
)

deploydocs(
    repo="github.com/JuliaFirstOrder/ProximalAlgorithms.jl.git",
)
