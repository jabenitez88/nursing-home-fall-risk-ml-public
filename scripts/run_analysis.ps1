$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

python src/article_pipeline.py --repeats 5 --folds 5 --bootstrap 500
