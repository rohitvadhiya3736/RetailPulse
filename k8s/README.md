# Kubernetes Deployment

These manifests deploy RetailPulse API, Streamlit dashboard, and a daily training CronJob.

## 1. Build Image

```bash
cd /Users/onlymec/RetailPulse
docker build -t rohitvadhiya3736/retailpulse:latest .
```

For Docker Desktop Kubernetes, the local image is usually available directly. For a remote cluster, push it:

```bash
docker push rohitvadhiya3736/retailpulse:latest
```

## 2. Deploy

```bash
kubectl apply -k k8s/
```

## 3. Check Status

```bash
kubectl get all -n retailpulse
kubectl logs -n retailpulse deploy/retailpulse-api
kubectl logs -n retailpulse deploy/retailpulse-dashboard
```

## 4. Open Locally

```bash
kubectl port-forward -n retailpulse svc/retailpulse-dashboard 8501:8501
kubectl port-forward -n retailpulse svc/retailpulse-api 8000:8000
kubectl port-forward -n retailpulse svc/retailpulse-prometheus 9090:9090
kubectl port-forward -n retailpulse svc/retailpulse-grafana 3000:3000
```

Dashboard: `http://localhost:8501`

API docs: `http://localhost:8000/docs`

Prometheus metrics: `http://localhost:8000/metrics`

Prometheus UI: `http://localhost:9090`

Grafana: `http://localhost:3000` with `admin` / `admin`

## 5. Run Training Manually

```bash
kubectl create job -n retailpulse --from=cronjob/retailpulse-daily-training retailpulse-training-manual
kubectl logs -n retailpulse job/retailpulse-training-manual -f
```

## 6. Delete

```bash
kubectl delete -k k8s/
```
