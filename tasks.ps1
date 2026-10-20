# PowerShell task runner for Windows development
param (
    [Parameter(Position=0)]
    [string]$Task = "help"
)

$env:Path = [Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [Environment]::GetEnvironmentVariable('Path','User')

switch ($Task) {
    "install" {
        python -m pip install --upgrade pip
        pip install -r requirements-dev.txt
        pip install -e .
    }
    "data" {
        python -m heart data
    }
    "train" {
        python -m heart train
    }
    "test" {
        pytest --cov=heart --cov=api --cov-report=term-missing
    }
    "lint" {
        ruff check .
    }
    "run-api" {
        uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
    }
    "docker-build" {
        docker build -t heart-disease-api:local .
    }
    "docker-up" {
        docker compose up -d
    }
    "docker-down" {
        docker compose down
    }
    "k8s-deploy" {
        minikube image load heart-disease-api:local
        helm upgrade --install heart-api ./helm/heart-api --set image.pullPolicy=Never
    }
    "k8s-clean" {
        helm uninstall heart-api
    }
    "all" {
        python -m heart data
        python -m heart train
        pytest --cov=heart --cov=api --cov-report=term-missing
    }
    default {
        Write-Host "Available Tasks in tasks.ps1:"
        Write-Host "  .\tasks.ps1 install       - Install packages and editable module"
        Write-Host "  .\tasks.ps1 data          - Clean and preprocess raw data"
        Write-Host "  .\tasks.ps1 train         - Train models and log with MLflow"
        Write-Host "  .\tasks.ps1 test          - Run pytest suite with coverage"
        Write-Host "  .\tasks.ps1 lint          - Run ruff linter"
        Write-Host "  .\tasks.ps1 run-api       - Run FastAPI server"
        Write-Host "  .\tasks.ps1 docker-build  - Build Docker image"
        Write-Host "  .\tasks.ps1 docker-up     - Start Docker Compose stack"
        Write-Host "  .\tasks.ps1 docker-down   - Stop Docker Compose stack"
        Write-Host "  .\tasks.ps1 k8s-deploy    - Deploy Helm chart to Minikube"
        Write-Host "  .\tasks.ps1 k8s-clean     - Uninstall Helm chart"
        Write-Host "  .\tasks.ps1 all           - Run data, train, and tests"
    }
}
