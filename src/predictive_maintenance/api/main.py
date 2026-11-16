from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from predictive_maintenance.api import dependencies
from predictive_maintenance.api.routes import router
from predictive_maintenance.inference.predictor import RULPredictor
from predictive_maintenance.services.simulation import simulation


@asynccontextmanager
async def lifespan(app: FastAPI):
    dependencies.predictor = RULPredictor(
        model_name="lstm",
        device="cpu",
    )

    try:
        yield
    finally:
        simulation.stop()
        dependencies.predictor = None


app = FastAPI(
    title="Predictive Maintenance API",
    description="NASA C-MAPSS FD001 RUL prediction service",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)

frontend_directory = Path(__file__).resolve().parents[3] / "frontend"
app.mount(
    "/dashboard",
    StaticFiles(directory=frontend_directory, html=True),
    name="dashboard",
)
