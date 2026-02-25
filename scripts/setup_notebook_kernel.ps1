$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Path $PSScriptRoot -Parent
Set-Location $repoRoot

Write-Host "Installing notebook dependencies from requirements-notebook.txt..."
python -m pip install -r requirements-notebook.txt

Write-Host "Registering Jupyter kernel: modelevaluationcharts..."
python -m ipykernel install --user --name modelevaluationcharts --display-name "Python (ModelEvaluationCharts)"

Write-Host "Installed kernels:"
python -m jupyter kernelspec list

Write-Host "Notebook environment setup complete."
