from collections.abc import Generator

from sqlalchemy.orm import Session

from predictive_maintenance.db.database import SessionLocal
from predictive_maintenance.inference.predictor import RULPredictor

predictor = None


def get_predictor() -> RULPredictor:
    if predictor is None:
        raise RuntimeError("Predictor is not initialized")

    return predictor


def get_db_session() -> Generator[Session, None, None]:
    with SessionLocal() as session:
        yield session
