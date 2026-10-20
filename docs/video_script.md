# Pipeline Video Recording Script & Checklist

**Assignment Requirement:** Include a short video recording (approx. 3-5 minutes) demonstrating the overall end-to-end MLOps pipeline.

**Presenter:** Deenadayalan C K (Student ID: 2025AE05402)  
**Project:** End-to-End MLOps Heart Disease Risk Prediction  
**Repository:** https://github.com/ckdeena-bits/mlops-heart-disease

---

## Pre-Recording Checklist
- [ ] Terminal open in repository directory with virtual environment active
- [ ] Browser (Firefox/Chrome) open with tabs ready:
  - Tab 1: GitHub Repository (`https://github.com/ckdeena-bits/mlops-heart-disease`)
  - Tab 2: GitHub Actions Workflow runs (`https://github.com/ckdeena-bits/mlops-heart-disease/actions`)
  - Tab 3: API Swagger Documentation (`http://localhost:8000/docs`)
  - Tab 4: MLflow Tracking UI (`http://localhost:5000`)
  - Tab 5: Grafana Monitoring Dashboard (`http://localhost:3000`)
- [ ] Screen recording tool (e.g. OBS Studio, SimpleScreenRecorder, or Xbox Game Bar) set to full HD (1920x1080).

---

## 3-Minute Walkthrough Script

### 0:00 - 0:30: Introduction & Architecture Overview
* **Action:** Show the GitHub repository README.md architecture diagram.
* **Narration:**  
  *"Hello, I am Deenadayalan C K, Student ID 2025AE05402. In this video, I will demonstrate the end-to-end MLOps pipeline for heart disease risk prediction developed for AIMLCZG523 Assignment 01. The solution covers automated data validation, multi-model experimentation with MLflow, unit testing in CI/CD, production Docker containerization, Kubernetes deployment with Helm, and Prometheus/Grafana monitoring."*

### 0:30 - 1:15: Code, Data Pipeline & Model Training
* **Action:** Switch to terminal, show repository structure, and execute `python -m heart data` then `python -m heart train`.
* **Narration:**  
  *"The repository follows production src layout. `python -m heart data` cleans and validates the 303 patient records against our schema constraints. `python -m heart train` runs 5-fold cross-validation with randomized hyperparameter search comparing Logistic Regression and Random Forest. Our champion model achieves an impressive ROC-AUC of 0.966 on the holdout test set, easily surpassing our production quality gate of 0.80."*

### 1:15 - 1:45: MLflow Experiment Tracking
* **Action:** Switch to MLflow UI in browser (`http://localhost:5000`). Click into both runs, show the comparison table, logged parameters, metrics, ROC curves, and the registered model artifact.
* **Narration:**  
  *"Here in the MLflow tracking UI, we can see both candidate models logged with parameters, cross-validation metrics, confusion matrices, and ROC curves. The champion model and its scikit-learn pipeline transformer are persisted as serialized artifacts."*

### 1:45 - 2:30: CI/CD Pipeline & Automated Testing
* **Action:** Show terminal running `pytest` (all 14 tests pass), then show GitHub Actions tab showing the multi-stage workflow: `lint-and-test` -> `train-and-track` -> `build-and-push-container`.
* **Narration:**  
  *"Our automated test suite verifies schema integrity, model performance gates, and API responses with 77% code coverage. The GitHub Actions CI/CD pipeline triggers on every push, executing Ruff linting, Pytest, automated retraining, and pushes the production container to GitHub Container Registry."*

### 2:30 - 3:15: GHCR Pull, Container Serving & Monitoring
* **Action:** 
  1. In terminal, run:
     ```bash
     docker pull ghcr.io/ckdeena-bits/mlops-heart-disease:latest
     ```
  2. Switch to browser showing `http://localhost:8000/docs`. Send a sample prediction via Swagger UI. Show the response returning `prediction`, `confidence`, and `risk_level: "Low Risk"`.
  3. Switch to Grafana dashboard (`http://localhost:3000`) showing live metrics.
* **Narration:**  
  *"Here, we demonstrate production reproducibility by pulling the pre-built, tested container image directly from GitHub Container Registry (ghcr.io). The FastAPI service serves inference via `/predict`, calculating real-time risk stratification bands. Operational metrics are scraped by Prometheus and displayed on our Grafana dashboard, showing request throughput, latency percentiles, and confidence distributions."*

### 3:15 - 3:45: Kubernetes Production Deployment & Conclusion
* **Action:** In terminal, show `kubectl get pods,svc,hpa` showing 2 healthy replicas and autoscaling configuration.
* **Narration:**  
  *"Finally, the container is packaged into a production Helm chart deployed on Kubernetes with health probes, horizontal pod autoscaling, and zero-downtime rolling upgrades. This concludes the demonstration. Thank you."*
