import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from predictive_maintenance.api.dependencies import get_db_session, get_predictor
from predictive_maintenance.db.repositories import (
    get_engine_history,
    get_engine_ids,
    get_prediction_history,
    save_prediction,
)
from predictive_maintenance.inference.predictor import RULPredictor
from predictive_maintenance.schemas.telemetry import (
    PredictionRequest,
    PredictionResponse,
)
from predictive_maintenance.services.reference_rul import reference_rul
from predictive_maintenance.services.sensor_monitor import sensor_overview
from predictive_maintenance.services.simulation import engine_catalog, simulation

router = APIRouter()
SENSOR_NAMES = {f"sensor_{number}" for number in range(1, 22)}


@router.get("/")
def root():
    return {"status": "running"}


@router.get("/health")
def health():
    return {"status": "healthy"}


@router.get("/engines")
def engines(session: Session = Depends(get_db_session)):
    catalog = engine_catalog()
    ids = sorted(set(get_engine_ids(session)) | {row["unit_id"] for row in catalog})
    return {"engines": ids, "catalog": catalog}


class SimulationRequest(BaseModel):
    unit_id: int = Field(gt=0)
    interval: float = Field(default=0.5, ge=0.1, le=5)


@router.get("/simulation")
def simulation_status():
    return simulation.snapshot()


@router.post("/simulation/start")
def start_simulation(data: SimulationRequest):
    try:
        return simulation.start(data.unit_id, data.interval)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.post("/simulation/stop")
def stop_simulation():
    return simulation.stop()


@router.get("/engines/{unit_id}/dashboard")
def engine_dashboard(
    unit_id: int,
    sensor: str = Query(default="sensor_4"),
    session: Session = Depends(get_db_session),
):
    if sensor not in SENSOR_NAMES:
        raise HTTPException(status_code=400, detail="Unknown sensor")

    telemetry = get_engine_history(session, unit_id)
    predictions = get_prediction_history(session, unit_id)
    state = simulation.snapshot()
    if state["unit_id"] == unit_id:
        telemetry = [row for row in telemetry if row["cycle"] <= state["cycle"]]
        predictions = [row for row in predictions if row["cycle"] <= state["cycle"]]
    latest_prediction = predictions[-1] if predictions else None
    current_cycle = telemetry[-1]["cycle"] if telemetry else 0
    prediction_reference = (
        reference_rul(unit_id, latest_prediction["cycle"]) if latest_prediction else None
    )

    return {
        "unit_id": unit_id,
        "current_cycle": current_cycle,
        "reference_rul": reference_rul(unit_id, current_cycle),
        "reference_rul_values": [reference_rul(unit_id, row["cycle"]) for row in telemetry],
        "prediction_reference_rul": prediction_reference,
        "prediction_error": latest_prediction["predicted_rul"] - prediction_reference
        if latest_prediction and prediction_reference is not None
        else None,
        "telemetry_cycles": [row["cycle"] for row in telemetry],
        "sensor_name": sensor,
        "sensor_values": [row[sensor] for row in telemetry],
        "prediction_cycles": [row["cycle"] for row in predictions],
        "predicted_rul_values": [row["predicted_rul"] for row in predictions],
        "latest_prediction": latest_prediction,
        "simulation": state,
        "overview": sensor_overview(telemetry),
        "sensor_series": {
            name: [row[name] for row in telemetry]
            for name in sorted(SENSOR_NAMES, key=lambda name: int(name.split("_")[1]))
        },
    }


@router.post("/engines/{unit_id}/predict", response_model=PredictionResponse)
def predict_engine(
    unit_id: int,
    cycle: int | None = Query(default=None, gt=0),
    predictor: RULPredictor = Depends(get_predictor),
    session: Session = Depends(get_db_session),
):
    measurements = get_engine_history(session, unit_id)
    if cycle is not None:
        measurements = [row for row in measurements if row["cycle"] <= cycle]
    else:
        state = simulation.snapshot()
        if state["unit_id"] == unit_id:
            measurements = [row for row in measurements if row["cycle"] <= state["cycle"]]
    if not measurements:
        raise HTTPException(status_code=404, detail="Engine telemetry not found")

    result = predictor.predict(pd.DataFrame(measurements)).iloc[0]
    prediction = {
        "unit_id": int(result["unit_id"]),
        "predicted_rul": float(result["predicted_rul"]),
        "history_cycles": int(result["history_cycles"]),
        "is_warmup": bool(result["is_warmup"]),
    }

    save_prediction(
        session,
        cycle=measurements[-1]["cycle"],
        model_name="lstm",
        **prediction,
    )

    return prediction


@router.post("/predict", response_model=list[PredictionResponse])
def predict(
    data: PredictionRequest,
    predictor: RULPredictor = Depends(get_predictor),
):
    df = pd.DataFrame([measurement.model_dump() for measurement in data.measurements])
    try:
        result = predictor.predict(df)
    except (ValueError, TypeError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return result.to_dict(orient="records")
