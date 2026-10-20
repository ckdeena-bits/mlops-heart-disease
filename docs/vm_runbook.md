# Rocky Linux 9.5 VM Deployment Runbook

**Environment:** Rocky Linux 9.5 (Blue Onyx), Python 3.12, Docker CE, Minikube, Helm  
**Author:** Deenadayalan C K (Student ID: 2025AE05402)  
**Repository:** `https://github.com/ckdeena-bits/mlops-heart-disease.git`

---

## 1. Prerequisites Installation (with `sudo`)

Run the following commands in the Rocky Linux terminal to ensure Docker, Minikube, and Helm are ready:

### A. Install Docker CE
```bash
sudo dnf config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
sudo dnf install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
# Log out and log back in, or run:
newgrp docker
```

### B. Install Minikube, kubectl & Helm
```bash
# kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Minikube
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
rm -f minikube-linux-amd64

# Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Verify installations
docker --version
kubectl version --client
minikube version
helm version
```

---

## 2. Clone Repository & Setup Virtual Environment

```bash
git clone https://github.com/ckdeena-bits/mlops-heart-disease.git
cd mlops-heart-disease

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -e .
```

---

## 3. Run Pipeline, Training & Tests

```bash
# 1. Clean & validate data
python -m heart data

# 2. Train models with 5-fold CV & MLflow
python -m heart train

# 3. View MLflow UI in Firefox (run quietly in background)
nohup mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000 > /dev/null 2>&1 &
# Open Firefox at: http://localhost:5000
# (To stop MLflow later: pkill -f "mlflow ui")

# 4. Run Pytest Suite with Coverage & Quality Gates
pytest --cov=heart --cov=api --cov-report=term-missing
```

---

## 4. Run Locally with Docker & Compose

```bash
# Start API + Prometheus + Grafana stack
docker compose up -d

# Check running services
docker compose ps

# Make a test prediction request via curl:
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}'

# Access dashboards in Firefox:
# - API Swagger Docs: http://localhost:8000/docs
# - Prometheus:       http://localhost:9090
# - Grafana:          http://localhost:3000 (User: admin / Pass: admin)

# Stop services when done:
docker compose down
```

---

## 5. Kubernetes Production Deployment with Minikube & Helm

```bash
# 1. Start Minikube with Docker driver
minikube start --driver=docker

# 2. Enable NGINX Ingress and Metrics Server addons
minikube addons enable ingress
minikube addons enable metrics-server

# 3. Build & load local image into Minikube
docker build -t heart-disease-api:local .
minikube image load heart-disease-api:local

# 4. Deploy Helm Chart
helm upgrade --install heart-api ./helm/heart-api --set image.pullPolicy=Never

# 5. Verify Pods, Service, and HPA
kubectl get pods
kubectl get svc
kubectl get hpa

# 6. Test Endpoint through Minikube Service Tunnel or Port-Forward
kubectl port-forward svc/heart-api 8080:80 &
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}'

# 7. Uninstall Helm chart when done
helm uninstall heart-api
```
