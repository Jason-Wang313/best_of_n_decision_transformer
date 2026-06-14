$ErrorActionPreference = "Stop"

$PaperDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $PaperDir
$BuildDir = Join-Path $RepoRoot "build\latex"
$FinalDir = Join-Path $PaperDir "final"
$FinalPdf = Join-Path $FinalDir "best_of_n_decision_transformer-v3.pdf"

New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
New-Item -ItemType Directory -Force -Path $FinalDir | Out-Null

Push-Location $RepoRoot
try {
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory $BuildDir paper/main.tex
    bibtex (Join-Path $BuildDir "main")
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory $BuildDir paper/main.tex
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory $BuildDir paper/main.tex
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory $BuildDir paper/main.tex

    $BuiltPdf = Join-Path $BuildDir "main.pdf"
    if (-not (Test-Path -LiteralPath $BuiltPdf)) {
        throw "Expected build output was not created: $BuiltPdf"
    }

    Copy-Item -LiteralPath $BuiltPdf -Destination $FinalPdf -Force
    Write-Host "Wrote $FinalPdf"
}
finally {
    Pop-Location
}
