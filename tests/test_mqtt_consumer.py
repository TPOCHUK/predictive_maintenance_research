import json
from contextlib import nullcontext
from types import SimpleNamespace

from predictive_maintenance.mqtt import consumer


def payload(**overrides):
    return dict(
        unit_id=11,
        cycle=5,
        **{f"setting_{i}": 0.0 for i in range(1, 4)},
        **{f"sensor_{i}": 1.0 for i in range(1, 22)},
        **overrides,
    )


def test_replay_still_requests_prediction_for_current_cycle(monkeypatch):
    history = [{"unit_id": 11, "cycle": cycle} for cycle in range(1, 84)]
    calls = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "unit_id": 11,
                "predicted_rul": 42.0,
                "history_cycles": 5,
                "is_warmup": True,
            }

    monkeypatch.setattr(consumer, "SessionLocal", lambda: nullcontext(object()))
    monkeypatch.setattr(consumer, "save_telemetry", lambda session, payload: False)
    monkeypatch.setattr(consumer, "get_engine_history", lambda session, unit_id: history)
    monkeypatch.setattr(
        consumer.requests,
        "post",
        lambda url, **kwargs: calls.append((url, kwargs)) or Response(),
    )

    message = SimpleNamespace(payload=json.dumps(payload(is_last=False)).encode())
    consumer.on_message(None, None, message)

    assert len(calls) == 1
    assert calls[0][1]["params"] == {"cycle": 5}


def test_final_cycle_always_requests_prediction(monkeypatch):
    history = [{"unit_id": 11, "cycle": cycle} for cycle in range(1, 84)]
    calls = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "unit_id": 11,
                "predicted_rul": 12.0,
                "history_cycles": 83,
                "is_warmup": False,
            }

    monkeypatch.setattr(consumer, "SessionLocal", lambda: nullcontext(object()))
    monkeypatch.setattr(consumer, "save_telemetry", lambda session, payload: False)
    monkeypatch.setattr(consumer, "get_engine_history", lambda session, unit_id: history)
    monkeypatch.setattr(
        consumer.requests,
        "post",
        lambda url, **kwargs: calls.append((url, kwargs)) or Response(),
    )

    message = SimpleNamespace(payload=json.dumps({**payload(is_last=True), "cycle": 83}).encode())
    consumer.on_message(None, None, message)

    assert len(calls) == 1
    assert calls[0][1]["params"] == {"cycle": 83}


def test_invalid_message_does_not_reach_database(monkeypatch):
    calls = []
    monkeypatch.setattr(consumer, "SessionLocal", lambda: calls.append("opened"))
    message = SimpleNamespace(payload=json.dumps({**payload(), "sensor_1": "bad"}).encode())
    consumer.on_message(None, None, message)
    assert calls == []
