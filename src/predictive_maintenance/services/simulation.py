"""Browser-controlled replay. Existing database rows are preserved on replay."""

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from threading import Event, Lock, Thread

import paho.mqtt.client as mqtt
import pandas as pd

MQTT_BROKER_HOST = os.getenv("MQTT_BROKER_HOST", "localhost")
MQTT_BROKER_PORT = int(os.getenv("MQTT_BROKER_PORT", "1883"))


def connect_mqtt() -> mqtt.Client:
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, keepalive=60)
    client.loop_start()
    return client


@lru_cache(maxsize=1)
def dataset() -> pd.DataFrame:
    columns = ["unit_id", "cycle"] + [f"setting_{i}" for i in range(1, 4)]
    columns += [f"sensor_{i}" for i in range(1, 22)]
    path = Path(__file__).resolve().parents[3] / "data" / "test_FD001.txt"
    return pd.read_csv(path, sep=r"\s+", header=None, names=columns)


def engine_catalog() -> list[dict]:
    return [
        {"unit_id": int(unit), "total_cycles": int(size)}
        for unit, size in dataset().groupby("unit_id").size().items()
    ]


class Simulation:
    def __init__(self):
        self._lock = Lock()
        self._stop = Event()
        self._thread = None
        self._state = {
            "status": "idle",
            "unit_id": None,
            "cycle": 0,
            "total_cycles": 0,
            "error": None,
        }

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._state)

    def start(self, unit_id: int, interval: float) -> dict:
        rows = dataset().loc[lambda df: df.unit_id == unit_id].copy()
        if rows.empty:
            raise ValueError("Engine not found in dataset")
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("A simulation is already running; stop it first")
            self._stop = Event()
            self._state = {
                "status": "running",
                "unit_id": unit_id,
                "cycle": 0,
                "total_cycles": len(rows),
                "error": None,
            }
            self._thread = Thread(
                target=self._run, args=(rows, interval), daemon=True, name="engine-simulation"
            )
            self._thread.start()
            return dict(self._state)

    def stop(self) -> dict:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=10)
        return self.snapshot()

    def _run(self, rows, interval):
        """Надсилаємо цикли через MQTT. Consumer зберігає їх і просить прогноз."""
        client = None
        try:
            client = connect_mqtt()
            for index in range(len(rows)):
                if self._stop.is_set():
                    break
                payload = rows.iloc[index].to_dict()
                payload["unit_id"] = int(payload["unit_id"])
                payload["cycle"] = int(payload["cycle"])
                payload["is_last"] = index == len(rows) - 1
                topic = f"engines/{payload['unit_id']}/telemetry"
                publish = client.publish(topic, json.dumps(payload), qos=1)
                publish.wait_for_publish(timeout=10)
                if not publish.is_published():
                    raise RuntimeError("MQTT publish timed out")
                with self._lock:
                    self._state["cycle"] = payload["cycle"]
                if self._stop.wait(interval):
                    break
            with self._lock:
                self._state["status"] = "stopped" if self._stop.is_set() else "completed"
        except Exception:
            logging.getLogger(__name__).exception("Simulation failed")
            with self._lock:
                self._state.update(
                    status="error", error="Simulation failed. Check API terminal logs."
                )
        finally:
            if client is not None:
                client.loop_stop()
                client.disconnect()


simulation = Simulation()
