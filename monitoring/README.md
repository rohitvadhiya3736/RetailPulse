# Prometheus and Grafana Monitoring

RetailPulse exposes Prometheus metrics from the FastAPI service at:

```text
http://localhost:8000/metrics
```

Metrics include:

- `retailpulse_http_requests_total`
- `retailpulse_http_request_duration_seconds`
- `retailpulse_model_metric{model="forecasting",metric="mape"}`
- `retailpulse_model_metric{model="churn",metric="roc_auc"}`

## Run Locally

```bash
cd /Users/onlymec/RetailPulse
docker compose up retailpulse-api prometheus grafana
```

Open:

- API docs: `http://localhost:8000/docs`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`

Grafana login:

```text
username: admin
password: admin
```

The RetailPulse dashboard is provisioned automatically under the `RetailPulse` folder.

## Test Metrics

Call the API a few times:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/metrics
curl http://localhost:8000/metrics
```

Then open Grafana and check:

- API request rate
- p95 latency
- Forecast MAPE
- Churn ROC-AUC
- 5xx error rate

