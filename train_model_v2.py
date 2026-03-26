#!/usr/bin/env python3
"""
CFL Play Predictor — Model Training Script v2
==============================================
Improvements over v1:
  1. Game-based train/test split (prevents correlated-play leakage)
  2. Opponent (defensive_team) added as feature
  3. CFL-specific interaction features (down × yards, situational flags)
  4. HistGradientBoostingClassifier replaces RandomForest
     — better calibrated probabilities, handles interactions natively
  5. Score differential computed correctly from home/away parse

Usage:
    python train_model_v2.py --data CFL_PLAY_BY_PLAY.xlsx --out ./

Outputs:
    cfl_play_predictor_model_v2.pkl  — model package (model + encoders + feature list)
    cfl_model_metrics_v2.json        — evaluation metrics and metadata
    cfl_historical_comparables_v2.csv — updated comparables with new features
    cfl_team_bucket_lookup_v2.csv    — unchanged bucket lookup (still valid)
"""

import argparse
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, classification_report, log_loss, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD & FILTER
# ─────────────────────────────────────────────────────────────────────────────
def load_and_filter(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = df[df["play_type"].isin(["Pass", "Run", "Sack"])].copy()

    # Target: called_pass = 1 if the offense intended to pass (pass or sack)
    df["called_pass"] = np.where(df["play_type"].isin(["Pass", "Sack"]), 1, 0)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. SCORE DIFFERENTIAL (offense perspective)
# ─────────────────────────────────────────────────────────────────────────────
def add_score_diff(df: pd.DataFrame) -> pd.DataFrame:
    # Game column format: "WEEK X - AWAY @ HOME"
    df["away_team"] = df["game"].str.extract(r"- (\w+) @")
    df["home_team"] = df["game"].str.extract(r"@ (\w+)")
    df["score_diff_offense"] = np.where(
        df["possession_team"] == df["home_team"],
        df["home_team_score"] - df["away_team_score"],
        df["away_team_score"] - df["home_team_score"],
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Fill missing values in core features
    df["yards_to_go"]               = df["yards_to_go"].fillna(df["yards_to_go"].median())
    df["yards_to_endzone"]          = df["yards_to_endzone"].fillna(df["yards_to_endzone"].median())
    df["seconds_in_half_remaining"] = df["seconds_in_half_remaining"].fillna(
        df["seconds_in_half_remaining"].median()
    )
    df["defensive_team"] = df["defensive_team"].fillna("UNK")

    # ── CFL-Specific Interactions ──────────────────────────────────────────
    # In a 3-down league, down × yards is a much stronger signal than either alone.
    # 2nd-and-long (≥8) is the CFL equivalent of NFL 3rd-and-long — almost always pass.
    # 2nd-and-short (≤3) is a common run situation.
    df["down_x_yards"] = df["down"] * df["yards_to_go"]
    df["is_2nd_short"] = ((df["down"] == 2) & (df["yards_to_go"] <= 3)).astype(int)

    # ── Game Situation Flags ───────────────────────────────────────────────
    df["garbage_time"]    = (
        (df["score_diff_offense"].abs() > 20) &
        (df["seconds_in_half_remaining"] < 600)
    ).astype(int)
    df["leading_late"]    = (
        (df["score_diff_offense"] > 7) &
        (df["seconds_in_half_remaining"] < 300)
    ).astype(int)
    df["two_min_trailing"] = (
        (df["seconds_in_half_remaining"] <= 120) &
        (df["score_diff_offense"] < 0)
    ).astype(int)

    # ── Encode Categorical Teams ───────────────────────────────────────────
    le_off = LabelEncoder()
    le_def = LabelEncoder()
    df["possession_team_enc"] = le_off.fit_transform(df["possession_team"])
    df["defensive_team_enc"]  = le_def.fit_transform(df["defensive_team"])

    df._le_offense = le_off
    df._le_defense = le_def

    return df


FEATURES = [
    # Identity
    "possession_team_enc",     # Which team is on offense
    "defensive_team_enc",      # Which team is defending (NEW vs v1)
    # Situation
    "quarter",
    "down",
    "yards_to_go",
    "yards_to_endzone",
    "seconds_in_half_remaining",
    "score_diff_offense",
    # Interactions (NEW vs v1)
    "down_x_yards",            # Primary CFL signal: captures 3-down pressure
    "is_2nd_short",            # 2nd-and-short run tendency
    "garbage_time",            # Blowout late — strategy changes
    "leading_late",            # Late-game run-heavy tendency
    "two_min_trailing",        # Two-minute drill — pass-heavy
]


# ─────────────────────────────────────────────────────────────────────────────
# 4. GAME-BASED TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────
def game_split(df: pd.DataFrame, test_fraction: float = 0.25):
    """
    Hold out the last `test_fraction` of season games.
    
    WHY: Plays within the same game are correlated — splitting randomly
    puts plays from the same game in both train and test, inflating accuracy.
    A game-based split reflects real deployment: the model is trained on
    earlier games and evaluated on later ones.
    """
    game_ids = sorted(df["cfl_game_id"].unique())
    n_test = int(len(game_ids) * test_fraction)
    test_ids  = set(game_ids[-n_test:])
    train_ids = set(game_ids[:-n_test])

    train = df[df["cfl_game_id"].isin(train_ids)]
    test  = df[df["cfl_game_id"].isin(test_ids)]
    return train, test, train_ids, test_ids


# ─────────────────────────────────────────────────────────────────────────────
# 5. TRAIN
# ─────────────────────────────────────────────────────────────────────────────
def train_model(X_train, y_train):
    """
    HistGradientBoostingClassifier is sklearn's native gradient-boosted tree
    implementation (equivalent to XGBoost/LightGBM). It natively handles:
      - Missing values (no imputer needed)
      - Feature interactions via tree structure
      - Better probability calibration than RandomForest
    """
    model = HistGradientBoostingClassifier(
        max_iter=500,
        max_depth=6,
        learning_rate=0.04,
        min_samples_leaf=12,
        l2_regularization=0.5,
        class_weight="balanced",   # Corrects for 65/35 pass/run imbalance
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=25,
    )
    model.fit(X_train, y_train)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 6. EVALUATE
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(model, X_test, y_test, feature_names):
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)
    ll  = log_loss(y_test, proba)
    cr  = classification_report(y_test, preds, target_names=["Run", "Pass"], output_dict=True)

    print(f"Accuracy : {acc:.4f}")
    print(f"ROC-AUC  : {auc:.4f}  ← primary metric (threshold-independent)")
    print(f"Log Loss : {ll:.4f}  ← measures probability calibration quality")
    print()
    print(classification_report(y_test, preds, target_names=["Run", "Pass"]))

    # Permutation importance (model-agnostic, reliable for HGB)
    perm = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )
    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": perm.importances_mean,
        "std": perm.importances_std,
    }).sort_values("importance", ascending=False)
    print("Feature Importances (permutation):")
    print(imp_df.to_string(index=False))

    return {"accuracy": acc, "roc_auc": auc, "log_loss": ll, "report": cr, "importances": imp_df}


# ─────────────────────────────────────────────────────────────────────────────
# 7. BUILD COMPARABLES (updated with new features)
# ─────────────────────────────────────────────────────────────────────────────
def build_comparables(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "possession_team", "quarter", "down", "yards_to_go",
        "yards_to_endzone", "seconds_in_half_remaining", "score_diff_offense",
        "called_pass", "play_result", "description", "cfl_game_id", "play_id",
    ]
    # Add bucket labels for the lookup system
    df["distance_bucket"] = pd.cut(
        df["yards_to_go"],
        bins=[0, 3, 7, 99],
        labels=["Short (1-3)", "Medium (4-7)", "Long (8+)"],
    ).astype(str)
    df["field_bucket"] = pd.cut(
        df["yards_to_endzone"],
        bins=[0, 20, 40, 60, 80, 999],
        labels=["Red Zone", "Opponent Territory", "Midfield", "Own Territory", "Own Deep"],
    ).astype(str)
    df["score_bucket"] = pd.cut(
        df["score_diff_offense"],
        bins=[-999, -8, -1, 0, 7, 999],
        labels=["Trailing 8+", "Trailing 1-7", "Tied", "Leading 1-7", "Leading 8+"],
    ).astype(str)
    df["time_bucket"] = pd.cut(
        df["seconds_in_half_remaining"],
        bins=[-1, 120, 600, 9999],
        labels=["2-min", "Late Half", "Early Half"],
    ).astype(str)
    return df[cols + ["distance_bucket", "field_bucket", "score_bucket", "time_bucket"]]


# ─────────────────────────────────────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main(data_path: str, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = load_and_filter(data_path)
    df = add_score_diff(df)
    df = engineer_features(df)

    le_off = df._le_offense
    le_def = df._le_defense

    df_clean = df.dropna(subset=FEATURES + ["called_pass", "cfl_game_id"])
    print(f"Modeling rows: {len(df_clean)} | Pass rate: {df_clean['called_pass'].mean():.3f}")

    train, test, train_ids, test_ids = game_split(df_clean)
    X_train = train[FEATURES]; y_train = train["called_pass"]
    X_test  = test[FEATURES];  y_test  = test["called_pass"]

    print(f"Train: {len(train)} plays / {len(train_ids)} games")
    print(f"Test:  {len(test)} plays / {len(test_ids)} games")

    print("\nTraining model...")
    model = train_model(X_train, y_train)

    print("\nEvaluation:")
    results = evaluate(model, X_test, y_test, FEATURES)

    # Save model package
    model_package = {
        "model": model,
        "le_offense": le_off,
        "le_defense": le_def,
        "features": FEATURES,
        "offense_teams": le_off.classes_.tolist(),
        "defense_teams": le_def.classes_.tolist(),
    }
    with open(out / "cfl_play_predictor_model_v2.pkl", "wb") as f:
        pickle.dump(model_package, f)

    # Save metrics
    metrics = {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_games": int(len(train_ids)),
        "test_games": int(len(test_ids)),
        "overall_rows_modeled": int(len(df_clean)),
        "accuracy_at_0_5_threshold": float(results["accuracy"]),
        "roc_auc": float(results["roc_auc"]),
        "log_loss": float(results["log_loss"]),
        "pass_rate_overall": float(df_clean["called_pass"].mean()),
        "classification_report": results["report"],
        "model_type": "HistGradientBoostingClassifier (sklearn — equivalent to XGBoost)",
        "model_parameters": {
            "max_iter": 500, "max_depth": 6, "learning_rate": 0.04,
            "min_samples_leaf": 12, "l2_regularization": 0.5,
            "class_weight": "balanced", "early_stopping": True,
        },
        "features_used": FEATURES,
        "target_definition": "called_pass = 1 if pass/sack (intent-based), 0 if run",
        "split_method": "game-based — last 25% of season games held out (no leakage)",
        "v1_comparison": {
            "v1_accuracy": 0.7154,
            "v1_split": "random rows (inflated — correlated plays leaked between train/test)",
            "v2_accuracy": float(results["accuracy"]),
            "v2_roc_auc": float(results["roc_auc"]),
            "v2_log_loss": float(results["log_loss"]),
            "key_improvements": [
                "Game-based split: no correlated-play leakage",
                "Opponent (defensive_team) as feature",
                "down × yards interaction (critical for CFL 3-down system)",
                "Situational flags: garbage_time, leading_late, two_min_trailing, is_2nd_short",
                "HistGradientBoosting: better calibrated probabilities than RandomForest",
            ],
        },
    }
    with open(out / "cfl_model_metrics_v2.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save comparables
    comps = build_comparables(df_clean.copy())
    comps.to_csv(out / "cfl_historical_comparables_v2.csv", index=False)

    print(f"\n✅ All outputs saved to {out}/")
    print(f"   cfl_play_predictor_model_v2.pkl")
    print(f"   cfl_model_metrics_v2.json")
    print(f"   cfl_historical_comparables_v2.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CFL play predictor v2")
    parser.add_argument("--data", default="CFL_PLAY_BY_PLAY.xlsx", help="Path to play-by-play Excel file")
    parser.add_argument("--out",  default="./",                    help="Output directory")
    args = parser.parse_args()
    main(args.data, args.out)
