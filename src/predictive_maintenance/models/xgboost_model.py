from xgboost import XGBRegressor

SEED = 42

XGB_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.012074507989018332,
    "max_depth": 4,
    "min_child_weight": 3.2298041995296414,
    "subsample": 0.7900204897773602,
    "colsample_bytree": 0.9223090698240903,
    "reg_alpha": 0.7459635164536705,
    "reg_lambda": 4.416722489003329,
    "gamma": 0.2695542352154229,
    "objective": "reg:squarederror",
    "random_state": SEED,
    "n_jobs": -1,
    "tree_method": "hist",
}

xgb_model = XGBRegressor(**XGB_PARAMS)
