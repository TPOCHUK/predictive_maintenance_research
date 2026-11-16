from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from predictive_maintenance.db.models import Prediction, Telemetry

TELEMETRY_FIELDS = (
    "unit_id",
    "cycle",
    *(f"setting_{number}" for number in range(1, 4)),
    *(f"sensor_{number}" for number in range(1, 22)),
)


def save_telemetry(session: Session, payload: dict) -> bool:
    # MQTT can deliver the same cycle more than once.
    statement = (
        insert(Telemetry)
        .values(**payload)
        .on_conflict_do_nothing(
            index_elements=["unit_id", "cycle"],
        )
    )

    result = session.execute(statement)
    session.commit()

    return result.rowcount == 1


def get_engine_history(session: Session, unit_id: int) -> list[dict]:
    statement = select(Telemetry).where(Telemetry.unit_id == unit_id).order_by(Telemetry.cycle)

    rows = session.scalars(statement).all()

    return [{field: getattr(row, field) for field in TELEMETRY_FIELDS} for row in rows]


def get_engine_ids(session: Session) -> list[int]:
    statement = select(Telemetry.unit_id).distinct().order_by(Telemetry.unit_id)
    return list(session.scalars(statement))


def save_prediction(
    session: Session,
    *,
    unit_id: int,
    cycle: int,
    predicted_rul: float,
    history_cycles: int,
    is_warmup: bool,
    model_name: str = "lstm",
) -> None:
    # Replaying a cycle updates its prediction.
    values = {
        "unit_id": unit_id,
        "cycle": cycle,
        "predicted_rul": predicted_rul,
        "history_cycles": history_cycles,
        "is_warmup": is_warmup,
        "model_name": model_name,
    }

    statement = (
        insert(Prediction)
        .values(**values)
        .on_conflict_do_update(
            index_elements=["unit_id", "cycle", "model_name"],
            set_={
                "predicted_rul": predicted_rul,
                "history_cycles": history_cycles,
                "is_warmup": is_warmup,
                "created_at": func.now(),
            },
        )
    )

    session.execute(statement)
    session.commit()


def get_prediction_history(session: Session, unit_id: int) -> list[dict]:
    statement = select(Prediction).where(Prediction.unit_id == unit_id).order_by(Prediction.cycle)
    rows = session.scalars(statement).all()

    return [
        {
            "cycle": row.cycle,
            "predicted_rul": row.predicted_rul,
            "history_cycles": row.history_cycles,
            "is_warmup": row.is_warmup,
            "model_name": row.model_name,
            "created_at": row.created_at,
        }
        for row in rows
    ]
