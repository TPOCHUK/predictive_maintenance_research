from pydantic import BaseModel, ConfigDict, Field


class TelemetryInput(BaseModel):
    """Sensor data for one engine cycle."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    unit_id: int = Field(gt=0)
    cycle: int = Field(gt=0)

    setting_1: float
    setting_2: float
    setting_3: float

    sensor_1: float
    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_5: float
    sensor_6: float
    sensor_7: float
    sensor_8: float
    sensor_9: float
    sensor_10: float
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_14: float
    sensor_15: float
    sensor_16: float
    sensor_17: float
    sensor_18: float
    sensor_19: float
    sensor_20: float
    sensor_21: float


class PredictionRequest(BaseModel):
    """Sensor history for one or more engines."""

    measurements: list[TelemetryInput] = Field(min_length=1)


class PredictionResponse(BaseModel):
    """RUL prediction result."""

    unit_id: int
    predicted_rul: float
    history_cycles: int
    is_warmup: bool
