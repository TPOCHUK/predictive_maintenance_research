import json
import os

import paho.mqtt.client as mqtt
import requests

from predictive_maintenance.db.database import SessionLocal
from predictive_maintenance.db.repositories import (
    get_engine_history,
    save_telemetry,
)
from predictive_maintenance.schemas.telemetry import TelemetryInput

BROKER_HOST = os.getenv("MQTT_BROKER_HOST", "localhost")
BROKER_PORT = int(os.getenv("MQTT_BROKER_PORT", "1883"))
PREDICTION_INTERVAL = 5

TOPIC = "engines/+/telemetry"
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print("Connected to MQTT broker")
        client.subscribe(TOPIC)
        print(f"Subscribed to: {TOPIC}")
    else:
        print(f"MQTT connection failed: {reason_code}")


def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode("utf-8"))

        is_last = bool(payload.pop("is_last", False))
        payload = TelemetryInput.model_validate(payload).model_dump()
        unit_id = payload["unit_id"]
        cycle = payload["cycle"]

        with SessionLocal() as session:
            was_inserted = save_telemetry(session, payload)
            measurements = get_engine_history(session, unit_id)

        # Keep future rows from earlier replays out of this prediction.
        measurements = [row for row in measurements if row["cycle"] <= cycle]

        print(
            f"{'Stored' if was_inserted else 'Replayed'} engine {unit_id}, "
            f"cycle {cycle}, "
            f"total cycles: {len(measurements)}"
        )

        total_cycles = len(measurements)

        # The first prediction is ready after one cycle.
        if total_cycles != 1 and total_cycles % PREDICTION_INTERVAL != 0 and not is_last:
            return

        response = requests.post(
            f"{API_BASE_URL}/engines/{unit_id}/predict",
            params={"cycle": cycle},
            timeout=30,
        )

        response.raise_for_status()

        prediction = response.json()

        print(
            f"Engine {prediction['unit_id']} | "
            f"Predicted RUL: "
            f"{prediction['predicted_rul']:.2f} | "
            f"History: {prediction['history_cycles']} | "
            f"Warmup: {prediction['is_warmup']}"
        )

    except Exception as error:
        print(f"Error processing telemetry: {error}")


def main():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)

    print("Starting MQTT Consumer...")
    client.loop_forever()


if __name__ == "__main__":
    main()
