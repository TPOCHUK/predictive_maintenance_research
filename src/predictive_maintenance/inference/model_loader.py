import torch
import xgboost as xgb

from ..config import MODELS_DIR, ModelSpec, load_model_specs
from ..models.cnn import RULCNN
from ..models.lstm import RULLSTM
from ..models.xgboost_model import XGB_PARAMS


def load_model(spec: ModelSpec, device: str = "cpu") -> torch.nn.Module:
    if spec.architecture == "RULLSTM":
        model = RULLSTM(in_features=spec.input_size, hidden_size=spec.hidden_size)
    elif spec.architecture == "RULCNN":
        model = RULCNN(in_features=spec.input_size, window=spec.window_size, dropout_p=spec.dropout)
    else:
        raise ValueError(f"Unknown architecture: {spec.architecture}")
    # Load weights only, without running saved Python objects.
    state = torch.load(spec.weights, map_location=device, weights_only=True)
    if spec.checkpoint_key:
        state = state[spec.checkpoint_key]
    model.load_state_dict(state)
    return model.to(device).eval()


def load_lstm(device: str = "cpu"):
    return load_model(load_model_specs()["lstm"], device)


def load_cnn(device: str = "cpu"):
    return load_model(load_model_specs()["cnn"], device)


def load_xgboost():
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.load_model(MODELS_DIR / "best_xgboost_model.json")
    return model
