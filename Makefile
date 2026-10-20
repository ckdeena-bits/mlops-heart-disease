.PHONY: help install data train test lint run-api docker-build docker-up docker-down k8s-deploy k8s-clean all

help:
	@echo "Available commands:"
	@echo "  make install      - Install dev dependencies and package in editable mode"
	@echo "  make data         - Download and clean the UCI Heart Disease dataset"
	@echo "  make train        - Train models with CV and log experiments to MLflow"
	@echo "  make test         - Run pytest suite with code coverage"
	@echo "  make lint         - Run ruff linter checks"
	@echo "  make run-api      - Run FastAPI server locally on port 8000"
	@echo "  make docker-build - Build production Docker image"
	@echo "  make docker-up    - Start API, Prometheus, and Grafana via Docker Compose"
	@echo "  make docker-down  - Stop Docker Compose services"
	@echo "  make k8s-deploy   - Deploy Helm chart to Minikube"
	@echo "  make k8s-clean    - Uninstall Helm release from Minikube"
	@echo "  make all          - Run data cleaning, training, and test suite"

install:
	pip install --upgrade pip
	pip install -r requirements-dev.txt
	pip install -e .

data:
	python -m heart data

train:
	python -m heart train

test:
	pytest --cov=heart --cov=api --cov-report=term-missing

lint:
	ruff check .

run-api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t heart-disease-api:local .

docker-up:
	docker compose up -d

docker-down:
	docker compose down

k8s-deploy:
	minikube image load heart-disease-api:local
	helm upgrade --install heart-api ./helm/heart-api --set image.pullPolicy=Never

k8s-clean:
	helm uninstall heart-api

all: data train test
