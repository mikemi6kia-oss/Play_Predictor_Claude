#!/usr/bin/env python3
"""
precompute_tendencies.py
========================
Run this ONCE after training to pre-compute the Top 5 Tendencies table.
Add the output CSV to your Streamlit app folder.

Usage:
    python precompute_tendencies.py
"""

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent

# ─── Load model package ────────────────────────────────────────────────────────
with open(BASE / "cfl_play_predictor_model_v2.pkl", "rb") as f:
    pkg = pickle.load(f)

MODEL    = pkg["model"]
LE_OFF   = pkg["le_offense"]
LE_DEF   = pkg["le_defense"]
FEATURES = pkg["features"]

TEAM_LOOKUP = pd.read_csv(BASE / "cfl_team_bucket_lookup.csv")
TEAMS = sorted(TEAM_LOOKUP["possession_team"].dropna().unique().tolist())


def encode_team(le, team, fallback=0):
    return int(le.transform([team])[0]) if team in le.classes_ else fallback


def half_seconds(quarter, mins, secs):
    rem = int(mins) * 60 + int(secs)
    return rem + 900 if int(quarter) in (1, 3) else rem


def yards_to_endzone(side, yardline):
    return 110 - float(yardline) if side == "Own" else float(yardline)


def distance_bucket(x):
    if x <= 3:  return "Short (1-3)"
    if x <= 7:  return "Medium (4-7)"
    return "Long (8+)"

def field_bucket(x):
    if x <= 20: return "Red Zone"
    if x <= 40: return "Opponent Territory"
    if x <= 60: return "Midfield"
    if x <= 80: return "Own Territory"
    return "Own Deep"

def score_bucket(x):
    if x <= -8: return "Trailing 8+"
    if x <= -1: return "Trailing 1-7"
    if x == 0:  return "Tied"
    if x <= 7:  return "Leading 1-7"
    return "Leading 8+"

def time_bucket(sec):
    if sec <= 120: return "2-min"
    if sec <= 600: return "Late Half"
    return "Early Half"


def get_bucket(team, down, ytg, yte, score_diff, sec_half):
    tb  = distance_bucket(ytg); fb  = field_bucket(yte)
    sb  = score_bucket(score_diff); tmb = time_bucket(sec_half)
    sub = TEAM_LOOKUP[
        (TEAM_LOOKUP["possession_team"] == team) &
        (TEAM_LOOKUP["down"] == down) &
        (TEAM_LOOKUP["distance_bucket"] == tb) &
        (TEAM_LOOKUP["field_bucket"] == fb) &
        (TEAM_LOOKUP["score_bucket"] == sb) &
        (TEAM_LOOKUP["time_bucket"] == tmb)
    ]
    if sub.empty: return None
    return sub.sort_values("plays", ascending=False).iloc[0].to_dict()


def predict_pass_prob(team, def_team, quarter, mins, secs, down, ytg, side, ball_on, score_diff):
    yte      = yards_to_endzone(side, ball_on)
    sec_half = half_seconds(quarter, mins, secs)
    row = {
        "possession_team_enc":       encode_team(LE_OFF, team),
        "defensive_team_enc":        encode_team(LE_DEF, def_team),
        "quarter":                   quarter,
        "down":                      down,
        "yards_to_go":               ytg,
        "yards_to_endzone":          yte,
        "seconds_in_half_remaining": sec_half,
        "score_diff_offense":        score_diff,
        "down_x_yards":              down * ytg,
        "is_2nd_short":              int(down == 2 and ytg <= 3),
        "garbage_time":              int(abs(score_diff) > 20 and sec_half < 600),
        "leading_late":              int(score_diff > 7 and sec_half < 300),
        "two_min_trailing":          int(sec_half <= 120 and score_diff < 0),
    }
    x = pd.DataFrame([row])[FEATURES]
    return float(MODEL.predict_proba(x)[0, 1]), yte, sec_half


# ─── Grid search ───────────────────────────────────────────────────────────────
# Use league average opponent to isolate offensive tendency
GRID = dict(
    quarters     = [1, 2, 3, 4],
    downs        = [1, 2],
    minutes_list = [12, 9, 6, 3],
    seconds_list = [0],
    ytg_values   = [2.0, 5.0, 10.0],
    sides        = ["Own", "Opp"],
    ball_ons     = [15.0, 30.0, 45.0],
    score_diffs  = [-10, -3, 0, 3, 10],
)

rows = []
total = (len(TEAMS) * len(GRID["quarters"]) * len(GRID["downs"]) *
         len(GRID["minutes_list"]) * len(GRID["seconds_list"]) *
         len(GRID["ytg_values"]) * len(GRID["sides"]) *
         len(GRID["ball_ons"]) * len(GRID["score_diffs"]))
print(f"Computing {total:,} scenarios...")

count = 0
for team in TEAMS:
    # Use a neutral opponent for tendency isolation
    def_team = [t for t in TEAMS if t != team][0]
    for q in GRID["quarters"]:
     for d in GRID["downs"]:
      for mins in GRID["minutes_list"]:
       for secs in GRID["seconds_list"]:
        for ytg in GRID["ytg_values"]:
         for side in GRID["sides"]:
          for ball in GRID["ball_ons"]:
           for score in GRID["score_diffs"]:
            try:
                pp, yte, sec_half = predict_pass_prob(
                    team, def_team, q, mins, secs, d, ytg, side, ball, score
                )
                lk = get_bucket(team, d, ytg, yte, score, sec_half)
                if lk is None: continue
                delta = float(lk["pass_prob_delta_vs_league"])
                if abs(delta) >= 0.20 and int(lk["plays"]) >= 10 and int(lk["league_plays"]) >= 20:
                    rows.append({
                        "team": team, "quarter": q, "down": d,
                        "minutes": mins, "seconds": secs,
                        "yards_to_go": ytg, "field_side": side, "ball_on": ball,
                        "score_diff": score,
                        "model_pass_prob": pp,
                        "team_hist_pass_rate": float(lk["pass_prob_hist"]),
                        "league_pass_rate": float(lk["league_pass_prob_hist"]),
                        "delta_vs_league": delta,
                        "tendency": "Pass-heavy" if delta > 0 else "Run-heavy",
                        "team_plays": int(lk["plays"]),
                        "league_plays": int(lk["league_plays"]),
                    })
            except Exception:
                pass
            count += 1
    print(f"  {team} done ({count:,}/{total:,})")

df = pd.DataFrame(rows)
if df.empty:
    print("No tendencies found with current thresholds.")
else:
    df["abs_delta"] = df["delta_vs_league"].abs()
    df = df.sort_values(["team","abs_delta"], ascending=[True,False]).reset_index(drop=True)
    df["rank"] = df.groupby("team").cumcount() + 1
    top5 = df.groupby("team", group_keys=False).head(5)
    out_path = BASE / "cfl_top5_tendencies_precomputed.csv"
    top5.to_csv(out_path, index=False)
    print(f"\n✅ Saved {len(top5)} rows to {out_path}")
    print(top5[["team","tendency","delta_vs_league","down","yards_to_go","score_diff","team_plays"]].to_string())
