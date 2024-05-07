using Documenter, VectorizedReduction

makedocs(sitename = "VectorizedReduction.jl", pages = ["Home" => "index.md"])

deploydocs(
    repo = "github.com/andrewjradcliffe/VectorizedReduction.jl.git",
    branch = "gh-pages",
    devbranch = "main",
    versions = ["stable" => "v^", "v#.#", "dev" => "main"],
    devurl = "dev",
)
