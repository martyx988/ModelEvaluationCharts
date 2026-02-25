$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Path $PSScriptRoot -Parent
Set-Location $repoRoot

$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
  Write-Host "Creating project virtual environment (.venv)..."
  py -3.11 -m venv .venv
}

Write-Host "Installing notebook dependencies into .venv from requirements-notebook.txt..."
& $venvPython -m pip install -r requirements-notebook.txt

Write-Host "Registering Jupyter kernel: modelevaluationcharts-venv..."
& $venvPython -m ipykernel install --user --name modelevaluationcharts-venv --display-name "Python (ModelEvaluationCharts .venv)"

Write-Host "Installed kernels:"
& $venvPython -m jupyter kernelspec list

Write-Host "Notebook environment setup complete."
