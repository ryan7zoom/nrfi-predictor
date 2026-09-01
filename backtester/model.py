"""Model training: XGBoost with sample weighting, falls back to calibrated LogReg."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

FEATURE_COLUMNS = [
    "home_pitcher_season_fip", "home_pitcher_recent_fip_delta",
    "away_pitcher_season_fip", "away_pitcher_recent_fip_delta",
    "home_top3_woba", "away_top3_woba",
    "park_factor", "temperature_f", "wind_effect", "park_factor_x_wind",
    "home_lineup_stale_fraction", "away_lineup_stale_fraction",
]


def _feature_matrix(features_list: list[dict]) -> np.ndarray:
    rows = [[feat.get(col) if feat.get(col) is not None else 0.0 for col in FEATURE_COLUMNS]
            for feat in features_list]
    return np.array(rows, dtype=float)


def _labels(features_list: list[dict]) -> np.ndarray:
    return np.array([f.get("nrfi_label", 0) for f in features_list], dtype=int)


def _sample_weights(features_list: list[dict]) -> np.ndarray:
    return np.array([f.get("sample_weight", 1.0) for f in features_list], dtype=float)


def train_model(features_list: list[dict]):
    labeled = [f for f in features_list if f.get("nrfi_label") is not None]
    if len(labeled) < 10:
        return None

    X = _feature_matrix(labeled)
    y = _labels(labeled)
    weights = _sample_weights(labeled)

    try:
        import xgboost as xgb
        model = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, eval_metric="logloss")
        model.fit(X, y, sample_weight=weights)
        model._is_xgb = True
        return model
    except ImportError:
        base = LogisticRegression(max_iter=1000)
        model = CalibratedClassifierCV(base, cv=3)
        model.fit(X, y, sample_weight=weights)
        model._is_xgb = False
        return model


def predict(model, features_list: list[dict]) -> list[float]:
    if model is None:
        return [0.5] * len(features_list)
    X = _feature_matrix(features_list)
    return model.predict_proba(X)[:, 1].tolist()
