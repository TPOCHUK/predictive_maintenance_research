from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from predictive_maintenance.db.base import Base


class Telemetry(Base):
    __tablename__ = "telemetry"

    __table_args__ = (
        UniqueConstraint(
            "unit_id",
            "cycle",
            name="uq_telemetry_unit_cycle",
        ),
    )

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )

    unit_id: Mapped[int] = mapped_column(Integer, nullable=False)

    cycle: Mapped[int] = mapped_column(Integer, nullable=False)

    setting_1: Mapped[float] = mapped_column(Float, nullable=False)
    setting_2: Mapped[float] = mapped_column(Float, nullable=False)
    setting_3: Mapped[float] = mapped_column(Float, nullable=False)

    sensor_1: Mapped[float] = mapped_column(Float, nullable=False)
    sensor_2: Mapped[float] = mapped_column(Float, nullable=False)
    sensor_3: Mapped[float] = mapped_column(Float, nullable=False)
    sensor_4: Mapped[float] = mapped_column(Float, nullable=False)
    sensor_5: Mapped[float] = mapped_column(Float, nullable=False)
    sensor_6: Mapped[float] = mapped_column(Float, nullable=False)
    sensor_7: Mapped[float] = mapped_column(Float, nullable=False)
    sensor_8: Mapped[float] = mapped_column(Float, nullable=False)
    sensor_9: Mapped[float] = mapped_column(Float, nullable=False)
    sensor_10: Mapped[float] = mapped_column(Float, nullable=False)
    sensor_11: Mapped[float] = mapped_column(Float, nullable=False)
    sensor_12: Mapped[float] = mapped_column(Float, nullable=False)
    sensor_13: Mapped[float] = mapped_column(Float, nullable=False)
    sensor_14: Mapped[float] = mapped_column(Float, nullable=False)
    sensor_15: Mapped[float] = mapped_column(Float, nullable=False)
    sensor_16: Mapped[float] = mapped_column(Float, nullable=False)
    sensor_17: Mapped[float] = mapped_column(Float, nullable=False)
    sensor_18: Mapped[float] = mapped_column(Float, nullable=False)
    sensor_19: Mapped[float] = mapped_column(Float, nullable=False)
    sensor_20: Mapped[float] = mapped_column(Float, nullable=False)
    sensor_21: Mapped[float] = mapped_column(Float, nullable=False)

    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


class Prediction(Base):
    __tablename__ = "predictions"

    __table_args__ = (
        UniqueConstraint(
            "unit_id",
            "cycle",
            "model_name",
            name="uq_prediction_unit_cycle_model",
        ),
        Index(
            "ix_predictions_unit_created",
            "unit_id",
            "created_at",
        ),
    )

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )

    unit_id: Mapped[int] = mapped_column(Integer, nullable=False)

    cycle: Mapped[int] = mapped_column(Integer, nullable=False)

    predicted_rul: Mapped[float] = mapped_column(Float, nullable=False)

    history_cycles: Mapped[int] = mapped_column(Integer, nullable=False)

    is_warmup: Mapped[bool] = mapped_column(Boolean, nullable=False)

    model_name: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="lstm",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
