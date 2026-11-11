import datetime
import io
import logging
import os
import sys
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import Any, Optional

import joblib
import pandas as pd
import requests
from evidently import DataDefinition, Dataset, Regression, Report
from evidently.metrics import MAE, RMSE, R2Score
from evidently.presets import DataDriftPreset
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestRegressor

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Prometheus Metrics Definitions ---
registry = CollectorRegistry()

api_requests_total = Counter(
    "api_requests_total",
    "Total number of API requests",
    ["endpoint", "method", "status_code"],
    registry=registry,
)

api_request_duration_seconds = Histogram(
    "api_request_duration_seconds",
    "API request duration in seconds",
    ["endpoint", "method", "status_code"],
    registry=registry,
)

model_rmse_score = Gauge(
    "model_rmse_score",
    "Root mean squared error on the latest evaluation batch",
    registry=registry,
)

model_mae_score = Gauge(
    "model_mae_score",
    "Mean absolute error on the latest evaluation batch",
    registry=registry,
)

model_r2_score = Gauge(
    "model_r2_score",
    "R-squared score on the latest evaluation batch",
    registry=registry,
)

evidently_data_drift_detected_status = Gauge(
    "evidently_data_drift_detected_status",
    "Whether Evidently detected data drift: 1=yes, 0=no",
    registry=registry,
)

# --- Global Variables for Model and Data ---
TARGET = "cnt"
PREDICTION = "prediction"
NUM_FEATS = ["temp", "atemp", "hum", "windspeed", "mnth", "hr", "weekday"]
CAT_FEATS = ["season", "holiday", "workingday", "weathersit"]
ALL_MODEL_FEATS = NUM_FEATS + CAT_FEATS

DTEDAY_COL_NAME = "dteday"
COLUMNS_FOR_EVALUATION_PAYLOAD = ALL_MODEL_FEATS + [TARGET, DTEDAY_COL_NAME]

MODEL_PATH = Path("/app/artifacts/model.joblib")

# --- Data Ingestion and Preparation Functions ---
DATASET_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"


# --- Fonctions d'ingestion et de préparation des données (alignées sur l'examen Evidently) ---
def _fetch_data() -> pd.DataFrame:
    """Fetches the bike sharing dataset and returns a DataFrame."""
    logger.info("Fetching data from UCI archive...")
    try:
        content = requests.get(DATASET_URL, timeout=60).content
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            df = pd.read_csv(
                z.open("hour.csv"), header=0, sep=",", parse_dates=[DTEDAY_COL_NAME]
            )
        logger.info("Data fetched successfully.")
        return df
    except requests.exceptions.RequestException:
        logger.exception("Error fetching data. Check URL or network connection.")
        sys.exit(1)
    except Exception:
        logger.exception("Error processing fetched data.")
        sys.exit(1)


def _process_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Processes raw data, setting a DatetimeIndex as in the exam script."""
    logger.info("Processing raw data...")
    raw_data["hr"] = raw_data["hr"].astype(int)
    raw_data.index = raw_data.apply(
        lambda row: datetime.datetime.combine(
            row[DTEDAY_COL_NAME].date(), datetime.time(row.hr)
        ),
        axis=1,
    )
    raw_data = raw_data.sort_index()
    logger.info("Data processed successfully.")
    return raw_data


def _train_and_predict_reference_model(
    df: pd.DataFrame,
) -> tuple[RandomForestRegressor, pd.DataFrame]:
    """Train on January 2011 and add predictions to the reference data."""
    reference_data = df.loc["2011-01"].copy()
    if reference_data.empty:
        raise ValueError("No January 2011 data available for training.")

    # Use only model features, excluding the target and other dataset columns.
    features = reference_data[ALL_MODEL_FEATS]
    target = reference_data[TARGET]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(features, target)

    # These are training-set predictions for Evidently's reference baseline.
    reference_data[PREDICTION] = model.predict(features)
    return model, reference_data


@asynccontextmanager
async def lifespan(app: FastAPI):
    force_train = os.getenv("TRAIN", "false").lower() == "true"

    if force_train or not MODEL_PATH.is_file():
        logger.info("Training model")
        raw_data = _fetch_data()
        processed_data = _process_data(raw_data)
        model, reference_data = _train_and_predict_reference_model(processed_data)
        logger.info(
            "Training completed using %d January 2011 rows", len(reference_data)
        )
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump((model, reference_data), MODEL_PATH)
        logger.info("Model and reference data saved to %s", MODEL_PATH)
    else:
        logger.info("Loading saved model")
        model, reference_data = joblib.load(MODEL_PATH)
        logger.info(
            "Model and reference data loaded from %s (%d reference rows)",
            MODEL_PATH,
            len(reference_data),
        )

    app.state.model = model
    app.state.reference_data = reference_data

    yield


# --- FastAPI App Initialization ---
app = FastAPI(
    title="Bike Sharing Predictor API",
    description="API for predicting bike sharing demand with MLOps monitoring.",
    version="1.0.0",
    lifespan=lifespan,
)


# --- Pydantic Models for API Input/Output ---
class BikeSharingInput(BaseModel):
    temp: float = Field(..., example=0.24)
    atemp: float = Field(..., example=0.2879)
    hum: float = Field(..., example=0.81)
    windspeed: float = Field(..., example=0.0)
    mnth: int = Field(..., example=1)
    hr: int = Field(..., example=0)
    weekday: int = Field(..., example=6)
    season: int = Field(..., example=1)
    holiday: int = Field(..., example=0)
    workingday: int = Field(..., example=0)
    weathersit: int = Field(..., example=1)
    dteday: datetime.date = Field(
        ...,
        example="2011-01-01",
        description="Date of the record in YYYY-MM-DD format.",
    )


class PredictionOutput(BaseModel):
    predicted_count: float = Field(..., example=16.0)


class EvaluationData(BaseModel):
    data: list[dict[str, Any]] = Field(
        ...,
        description="List of data points, each containing features and the true target ('cnt').",
    )
    evaluation_period_name: str = Field(
        "unknown_period",
        description="Name of the period being evaluated (e.g., 'week1_february').",
    )
    model_config = {"arbitrary_types_allowed": True}


class EvaluationReportOutput(BaseModel):
    message: str
    rmse: Optional[float]
    mape: Optional[float]
    mae: Optional[float]
    r2score: Optional[float]
    drift_detected: int
    evaluated_items: int


# --- API Endpoints ---
@app.get("/")
async def read_root():
    return {
        "message": "Welcome to the Bike Sharing Predictor API. Use /predict to get bike counts or /evaluate to run drift reports."
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(data: BikeSharingInput, request: Request):
    features = pd.DataFrame([data.model_dump()])[ALL_MODEL_FEATS]
    prediction = request.app.state.model.predict(features)[0]
    return PredictionOutput(predicted_count=float(prediction))


@app.post("/evaluate", response_model=EvaluationReportOutput)
def evaluate(data: EvaluationData, request: Request):
    if not data.data:
        raise HTTPException(status_code=400, detail="Evaluation data is empty")

    current_data = pd.DataFrame(data.data)
    required_columns = ALL_MODEL_FEATS + [TARGET]
    missing = set(required_columns) - set(current_data.columns)

    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing columns: {sorted(missing)}",
        )

    current_data[PREDICTION] = request.app.state.model.predict(
        current_data[ALL_MODEL_FEATS]
    )

    definition = DataDefinition(
        numerical_columns=NUM_FEATS,
        categorical_columns=CAT_FEATS,
        regression=[Regression(target=TARGET, prediction=PREDICTION)],
    )

    report_columns = ALL_MODEL_FEATS + [TARGET, PREDICTION]

    reference_dataset = Dataset.from_pandas(
        request.app.state.reference_data[report_columns],
        data_definition=definition,
    )

    current_dataset = Dataset.from_pandas(
        current_data[report_columns],
        data_definition=definition,
    )

    report = Report(
        [
            RMSE(),
            MAE(),
            R2Score(),
            DataDriftPreset(columns=ALL_MODEL_FEATS),
        ]
    )

    snapshot = report.run(
        current_data=current_dataset,
        reference_data=reference_dataset,
    )

    # Identify metrics by type rather than generated IDs or display names.
    results = {
        metric["config"]["type"].rsplit(":", 1)[-1]: metric
        for metric in snapshot.dict()["metrics"]
    }
    rmse = float(results["RMSE"]["value"])
    mae = float(results["MAE"]["value"]["mean"])
    r2score = float(results["R2Score"]["value"])
    drift_result = results["DriftedColumnsCount"]
    # Flag input drift when the drifted feature share reaches Evidently's threshold.
    drift_detected = int(
        drift_result["value"]["share"] >= drift_result["config"]["drift_share"]
    )

    model_rmse_score.set(rmse)
    model_mae_score.set(mae)
    model_r2_score.set(r2score)
    evidently_data_drift_detected_status.set(drift_detected)

    logger.info(
        "Evaluation completed for %s: rows=%d, RMSE=%.4f, MAE=%.4f, R2=%.4f, drift=%d",
        data.evaluation_period_name, len(current_data), rmse, mae, r2score, drift_detected,
    )
    return EvaluationReportOutput(
        message=f"Evaluation completed for {data.evaluation_period_name}",
        rmse=rmse,
        mae=mae,
        r2score=r2score,
        mape=None,  # MAPE is not calculated; input drift is our additional metric.
        drift_detected=drift_detected,
        evaluated_items=len(current_data),
    )


@app.get("/metrics")
def metrics():
    return Response(
        content=generate_latest(registry),
        headers={"Content-Type": CONTENT_TYPE_LATEST},
    )


@app.middleware("http")
async def record_request_metrics(request: Request, call_next):
    endpoint = request.url.path

    if endpoint not in {"/predict", "/evaluate"}:
        return await call_next(request)

    start = perf_counter()
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        labels = {
            "endpoint": endpoint,
            "method": request.method,
            "status_code": str(status_code),
        }
        api_requests_total.labels(**labels).inc()
        api_request_duration_seconds.labels(**labels).observe(perf_counter() - start)
