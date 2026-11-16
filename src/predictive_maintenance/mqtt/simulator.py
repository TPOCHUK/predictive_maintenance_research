import json
import os
import time
from pathlib import Path

import paho.mqtt.client as mqtt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_ROOT / "data" / "test_FD001.txt"

BROKER_HOST = os.getenv("MQTT_BROKER_HOST", "localhost")
BROKER_PORT = int(os.getenv("MQTT_BROKER_PORT", "1883"))

UNIT_ID = 49
INTERVAL_SECONDS = 0.2

COLUMNS = (
    ["unit_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def load_telemetry() -> pd.DataFrame:
    df = pd.read_csv(
        DATA_PATH,
        sep=r"\s+",
        header=None,
        names=COLUMNS,
    )

    return df[df["unit_id"] == UNIT_ID].reset_index(drop=True)


def main():
    df = load_telemetry()

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)

    client.loop_start()

    print(f"Connected to MQTT broker: {BROKER_HOST}:{BROKER_PORT}")
    print(f"Simulating engine {UNIT_ID}")
    print(f"Total cycles: {len(df)}")

    try:
        for _, row in df.iterrows():
            payload = row.to_dict()

            payload["unit_id"] = int(payload["unit_id"])
            payload["cycle"] = int(payload["cycle"])
            payload["is_last"] = payload["cycle"] == int(df["cycle"].iloc[-1])

            message = json.dumps(payload)

            topic = f"engines/{UNIT_ID}/telemetry"

            info = client.publish(topic, message, qos=1)

            info.wait_for_publish(timeout=10)
            if not info.is_published():
                raise RuntimeError("MQTT publish timed out")

            print(f"Published cycle {payload['cycle']}")

            time.sleep(INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\nSimulation stopped.")

    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
