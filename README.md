# Predictive Maintenance for Turbofan Engines

An end-to-end predictive maintenance system that estimates the remaining useful life (RUL) of turbofan engines from sensor telemetry. It turns NASA C-MAPSS data into a working monitoring flow with MQTT ingestion, LSTM inference, PostgreSQL history, a FastAPI API, and a live dashboard—helping maintenance teams identify degradation before simulated engine failure.

## Results

Evaluated on all **100 engines** in the NASA C-MAPSS FD001 test set using the last available 80-cycle window per engine:

| Model | MAE (cycles) | RMSE (cycles) | R² | NASA score |
| --- | ---: | ---: | ---: | ---: |
| **Conservative LSTM Q20 (best experiment)** | **9.913** | **15.060** | **0.869** | **466.107** |
| LSTM (runtime model) | 13.12 | 20.38 | 0.760 | 1,845.16 |
| CNN | 17.66 | 25.76 | 0.616 | 74,201.60 |

The Conservative LSTM Q20 is the best experimental model in this work. Its quantile-loss objective reduces costly RUL overestimation and produced the best raw-test accuracy and NASA score. It is documented in [`notebooks/11_conservative_lstm_q20.ipynb`](notebooks/11_conservative_lstm_q20.ipynb), but is not currently integrated into the runtime application; the deployed inference path uses the standard LSTM shown above.

The repository includes the dataset subset, preprocessing artifacts, and trained weights required to reproduce the runtime LSTM and CNN metrics with `python main.py`. Results use raw RUL targets; capped metrics are also printed by the script. The Q20 result comes from its separate training notebook and is summarized in the project report.

For a detailed description of the methodology, feature pipeline, model experiments, and evaluation, see the [project report (PDF)](output/pdf/rul_fd001_course_style_en.pdf).

## What it does

- Replays multivariate telemetry for 100 simulated turbofan engines over MQTT.
- Validates, stores, and processes readings into 80-cycle model windows.
- Predicts RUL with a PyTorch LSTM and persists prediction history.
- Displays RUL, reference values, and sensor drift in a browser dashboard.
- Runs as a four-service Docker Compose stack with migrations and health checks.

## Tech stack

Python 3.11+, PyTorch, FastAPI, PostgreSQL, SQLAlchemy, Alembic, MQTT/Mosquitto, Docker Compose, JavaScript, pytest, Ruff, and GitHub Actions.

## Quick start

Requirements: Docker Desktop with Docker Compose.

```powershell
Copy-Item .env.example .env
```

Set a local `POSTGRES_PASSWORD` in `.env`, then start the stack:

```powershell
docker compose up --build -d --wait
docker compose logs mqtt-consumer
```

Wait for `Subscribed to: engines/+/telemetry`, then open:

- Dashboard: [http://127.0.0.1:8000/dashboard/](http://127.0.0.1:8000/dashboard/)
- API documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

Select an engine in the dashboard and start the simulation. Stop the stack with:

```powershell
docker compose down
```

## Local development

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
docker compose up -d postgres mosquitto
alembic upgrade head
uvicorn predictive_maintenance.api.main:app --reload
```

In a second activated terminal:

```powershell
python -m predictive_maintenance.mqtt.consumer
```

## Quality checks

```powershell
ruff check src tests scripts main.py migrations
ruff format --check src tests scripts main.py migrations
pytest -q
node tests/frontend_simulation_smoke.cjs
python main.py
```

GitHub Actions runs linting, unit tests, the frontend smoke test, and an end-to-end Docker replay on every push and pull request.

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the system flow, component boundaries, and runtime constraints.

## Scope

This is a local portfolio/demo system built on simulated FD001 data, not a production safety system. It currently has no authentication or durable message retry, and the replay coordinator assumes one API worker.
