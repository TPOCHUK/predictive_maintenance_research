"""Convert the old detector file to JSON."""

import joblib

from predictive_maintenance.data.artifacts import DETECTOR_PKL, save_detector

print("saved:", save_detector(joblib.load(DETECTOR_PKL)))
