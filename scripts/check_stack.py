"""Check a running demo stack. Replays engine 1 and keeps its DB history."""

import json
import time
from urllib.request import Request, urlopen


def request(path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = Request(
        f"http://127.0.0.1:8000{path}",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=10) as response:
        return json.load(response)


def main():
    request("/simulation/start", {"unit_id": 1, "interval": 0.1})
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        dashboard = request("/engines/1/dashboard")
        state = dashboard["simulation"]
        if state["status"] == "error":
            raise RuntimeError(state["error"])
        prediction = dashboard["latest_prediction"]
        if state["status"] == "completed" and prediction and prediction["cycle"] == 31:
            assert dashboard["telemetry_cycles"] == list(range(1, 32))
            assert prediction["history_cycles"] == 31
            assert dashboard["reference_rul"] == 112
            print("MQTT -> PostgreSQL -> API -> LSTM: OK")
            return
        time.sleep(1)
    raise TimeoutError("Engine 1 did not finish with a saved final prediction")


if __name__ == "__main__":
    main()
