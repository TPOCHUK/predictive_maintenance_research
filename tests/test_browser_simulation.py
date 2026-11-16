import json

import pandas as pd
from fastapi.testclient import TestClient

from predictive_maintenance.api import routes
from predictive_maintenance.api.dependencies import get_db_session, get_predictor
from predictive_maintenance.api.main import app
from predictive_maintenance.config import load_pipeline_config
from predictive_maintenance.services import simulation as simulation_module
from predictive_maintenance.services.reference_rul import reference_rul
from predictive_maintenance.services.sensor_monitor import sensor_overview

client = TestClient(app)


def request(method, path, body=None):
    return client.request(method, path, json=body)


def telemetry(count=40):
    return [
        dict(
            unit_id=1,
            cycle=i + 1,
            **{
                f"sensor_{n}": (i % 2) * 0.2 + (10 if n == 4 and i >= 25 else 0)
                for n in range(1, 22)
            },
        )
        for i in range(count)
    ]


class Predictor:
    def predict(self, frame):
        return pd.DataFrame(
            [
                dict(
                    unit_id=1,
                    predicted_rul=42.0,
                    history_cycles=len(frame),
                    is_warmup=len(frame) < 80,
                )
            ]
        )


class Published:
    def wait_for_publish(self, timeout):
        return None

    def is_published(self):
        return True


class MqttClient:
    def __init__(self, published):
        self.published = published

    def publish(self, topic, payload, qos):
        self.published.append((topic, json.loads(payload), qos))
        return Published()

    def loop_stop(self):
        pass

    def disconnect(self):
        pass


def test_detector_waits_for_baseline_and_identifies_drifting_sensor():
    assert not sensor_overview(telemetry(19))["ready"]
    result = sensor_overview(telemetry())
    assert result["baseline_cycles"] == load_pipeline_config().baseline_cycles
    assert len(result["scores"]) == 40
    assert {event["sensor"] for event in result["events"]} == {"sensor_4"}
    assert result["events"][0]["cycle"] > 25
    constant = [{**row, **{f"sensor_{n}": 5 for n in range(1, 22)}} for row in telemetry()]
    assert sensor_overview(constant)["events"] == []


def test_replay_publishes_every_cycle_to_mqtt(monkeypatch):
    rows = telemetry(6)
    published = []
    monkeypatch.setattr(simulation_module, "dataset", lambda: pd.DataFrame(rows))
    monkeypatch.setattr(simulation_module, "connect_mqtt", lambda: MqttClient(published))
    sim = simulation_module.Simulation()
    sim.start(1, 0)
    sim._thread.join(timeout=5)
    assert sim.snapshot()["status"] == "completed"
    assert [message[1]["cycle"] for message in published] == [1, 2, 3, 4, 5, 6]
    assert {message[0] for message in published} == {"engines/1/telemetry"}
    assert {message[2] for message in published} == {1}
    assert [message[1]["is_last"] for message in published] == [False] * 5 + [True]


def test_dashboard_hides_future_cycles_during_replay_and_predict_saves(monkeypatch):
    monkeypatch.setattr(routes, "get_engine_history", lambda session, unit: telemetry(40))
    monkeypatch.setattr(routes, "get_prediction_history", lambda session, unit: [])
    monkeypatch.setattr(
        routes.simulation,
        "snapshot",
        lambda: dict(status="running", unit_id=1, cycle=5, total_cycles=40, error=None),
    )
    saved = []
    monkeypatch.setattr(routes, "save_prediction", lambda session, **row: saved.append(row))
    app.dependency_overrides[get_db_session] = lambda: object()
    app.dependency_overrides[get_predictor] = lambda: Predictor()
    try:
        response = request("GET", "/engines/1/dashboard")
        assert response.status_code == 200
        assert response.json()["telemetry_cycles"] == [1, 2, 3, 4, 5]
        assert len(response.json()["sensor_series"]) == 21
        assert response.json()["reference_rul"] == 138
        assert response.json()["reference_rul_values"] == [142, 141, 140, 139, 138]
        assert request("POST", "/engines/1/predict").json()["history_cycles"] == 5
        assert saved[0]["cycle"] == 5
        assert request("GET", "/dashboard/").status_code == 200
        assert (
            request("POST", "/simulation/start", {"unit_id": 1, "interval": -1}).status_code == 422
        )
    finally:
        app.dependency_overrides.clear()


def test_stop_and_conflicting_start(monkeypatch):
    monkeypatch.setattr(simulation_module, "dataset", lambda: pd.DataFrame(telemetry()))
    monkeypatch.setattr(simulation_module, "connect_mqtt", lambda: MqttClient([]))
    sim = simulation_module.Simulation()
    sim.start(1, 5)
    try:
        import pytest

        with pytest.raises(RuntimeError):
            sim.start(1, 5)
    finally:
        sim.stop()
    assert sim.snapshot()["status"] == "stopped"


def test_fd001_reference_rul_uses_terminal_label_and_cycle():
    # Engine 1: last observed cycle 31, official remaining life 112.
    assert reference_rul(1, 31) == 112
    assert reference_rul(1, 1) == 142
    assert reference_rul(1, 0) is None
    assert reference_rul(9999, 1) is None
