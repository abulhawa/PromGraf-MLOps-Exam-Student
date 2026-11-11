# Bike Sharing monitoring exam

A Random Forest predicts hourly bike rental counts. FastAPI exposes prediction,
evaluation and Prometheus metrics; Grafana provisions three dashboards and a drift alert.
Original assignment: [exam instructions](docs/exam-instructions.md).

## Run

Prerequisites: Linux/WSL, Docker Engine with Docker Compose v2, GNU Make, and internet
access to pull images and download the public UCI Bike Sharing dataset.

```sh
make
make evaluation
```

`make` builds and starts the API, Prometheus, Grafana and Node Exporter.
On first API startup, January 2011 data trains the model. The model and reference
DataFrame are saved together in `artifacts/model.joblib`, persisted through a bind
mount. Later startups load this file. Evaluation waits for API health, evaluates
February 1–7, 2011, then sends 100 prediction requests. Allow a few scrapes for charts.

- API documentation: http://localhost:8080/docs
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (local anonymous Admin access)
- Node Exporter: http://localhost:9100/metrics

Grafana provisions the Prometheus datasource, all three dashboards in Bike Sharing,
and the drift alert from versioned files. No pre-existing Docker volumes are needed.
Grafana is bound to localhost because login is disabled for this local exam.

```sh
TRAIN=true make              # Recreate API with forced retraining.
make                        # Return TRAIN to false and load the saved model.
make stop                   # Stop/remove containers; retain volumes and artifacts.
```

Model artifacts must come from this project: joblib loading is for trusted files.
The archive excludes generated artifacts, so a fresh examiner run trains its own model.

## Metrics and dashboards

`/predict` accepts the supplied feature schema. `/evaluate` accepts `data` (records
including `cnt`) and `evaluation_period_name`. It runs Evidently against January,
returns a summary, and updates gauges. `/metrics` exposes the custom registry.

- API Performance: request rate, P95 latency and 5xx error fraction by endpoint.
- Model Performance & Drift: RMSE, MAE, R² and input drift status.
- Infrastructure Overview: CPU, memory and disk usage from Node Exporter.

The extra metric `evidently_data_drift_detected_status` is 1 when at least 50% of
input features drift according to Evidently's configured tests, otherwise 0.
This helps identify changes in incoming data; it does not prove prediction quality
has degraded. MAPE is not calculated and is returned as null. The API metrics cover
`/predict` and `/evaluate`, excluding monitoring scrapes. Gauges reset on API restart
and show initial zero values until evaluation is run again. With no traffic,
rate/latency/error charts may be empty; a successful traffic batch with no 5xx
errors shows a zero error fraction. Predictions on January are an in-sample
reference baseline, not an unbiased held-out performance estimate.

Node Exporter under Docker Desktop measures its Linux environment, not Windows.
Evidently may warn about R² on tiny internal chart bins; the overall batch score
uses all evaluation rows. Python dependencies are pinned; pandas 3.0.6 avoids the
known Evidently incompatibility with pandas 3.1. Node Exporter's image currently
uses `latest`, so that image can change between builds.

## Alert demonstration

```sh
make fire-alert
# Check http://localhost:9090/alerts after roughly 1–2 minutes.
make recover
# Check that BikeApiDown becomes inactive after scraping resumes.
make evaluation
```

`fire-alert` intentionally stops bike-api to test Prometheus `BikeApiDown`:
`up{job="bike-api"} == 0` for one minute. Recovery restarts the API. Grafana's
`BikeSharingDataDrift` rule checks drift status > 0.5 every minute with no pending
period. February evaluation provides the drift demonstration. The rule was created
in Grafana's interface and exported for provisioning. Alerts are visible in their
respective interfaces; external email/Slack notification delivery is not configured.

Grafana UI edits do not update repository JSON automatically. Export edited
dashboards/rules back into `deployment/grafana/` before submitting changes.

## Verification (November 11, 2026)

An isolated copy with fresh volumes and no model artifact successfully trained,
provisioned all three dashboards and the Grafana alert, and exposed three healthy
Prometheus targets. Evaluation of 165 February-only rows returned RMSE 20.7234 and
drift detected; all 100 traffic requests succeeded. Forced retraining and subsequent
loading of saved artifacts were also verified. Successful traffic produced zero
error fractions in the API dashboard. The temporary test services were removed.
