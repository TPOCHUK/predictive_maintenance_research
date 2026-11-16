# Architecture

## System flow

```text
NASA C-MAPSS FD001
        |
        v
FastAPI replay service --> Mosquitto --> MQTT consumer
                                         |
                         +---------------+---------------+
                         |                               |
                         v                               v
                    PostgreSQL                    LSTM predictor
                  telemetry history         validation -> features
                         ^                  -> scaling -> 80-cycle window
                         |                               |
                         +---------- prediction ---------+
                         |
                         v
                 FastAPI REST API
                         |
                         v
                  Browser dashboard
```

## Components

| Area | Responsibility |
| --- | --- |
| `api/` | REST endpoints, dependency wiring, dashboard hosting, and application lifecycle |
| `mqtt/` | Telemetry publishing and consumption |
| `data/` | Schema validation, feature engineering, scaling, and sequence creation |
| `inference/` | Model loading and RUL prediction |
| `db/` | SQLAlchemy models, sessions, and repositories |
| `services/` | Replay coordination, reference RUL, and sensor summaries |
| `frontend/` | Dashboard UI and API integration |
| `migrations/` | PostgreSQL schema evolution |

## Runtime path

1. The replay service publishes one engine cycle to `engines/{unit_id}/telemetry`.
2. The consumer validates and upserts telemetry in PostgreSQL.
3. At configured checkpoints, the consumer requests an API prediction.
4. The predictor loads the engine history, builds the model input, and returns RUL.
5. The prediction is stored and exposed to the dashboard through REST endpoints.

The LSTM uses 27 inputs across an 80-cycle window: operating settings, sensor readings, and a degradation feature. Short histories are left-padded and marked as warm-up predictions.

## Deployment

Docker Compose runs four services: `app`, `postgres`, `mosquitto`, and `mqtt-consumer`. Alembic migrations run before the API starts; health checks gate dependent services.

## Current constraints

- Replay state is in process memory, so the API should run with one worker.
- MQTT delivery is not backed by a durable retry or dead-letter queue.
- Authentication and authorization are outside the current demo scope.
- Reference RUL is for evaluation and visualization only; it is never a model input.
