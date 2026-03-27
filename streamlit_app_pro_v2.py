import json
import base64
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

BASE = Path(__file__).resolve().parent

st.set_page_config(
    page_title="CFL Play Predictor",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

TEAM_COLORS = {
    "BC":  {"primary": "#E35205", "secondary": "#000000"},
    "CGY": {"primary": "#C8102E", "secondary": "#F5A800"},
    "EDM": {"primary": "#2C5234", "secondary": "#F5A800"},
    "HAM": {"primary": "#FFB81C", "secondary": "#000000"},
    "MTL": {"primary": "#002D62", "secondary": "#C8102E"},
    "OTT": {"primary": "#CC0000", "secondary": "#000000"},
    "SSK": {"primary": "#006241", "secondary": "#C8A84B"},
    "TOR": {"primary": "#4169E1", "secondary": "#FFFFFF"},
    "WPG": {"primary": "#003087", "secondary": "#7BAFD4"},
}
TEAM_NAMES = {
    "BC":  "BC Lions",           "CGY": "Calgary Stampeders",
    "EDM": "Edmonton Elks",      "HAM": "Hamilton Tiger-Cats",
    "MTL": "Montreal Alouettes", "OTT": "Ottawa REDBLACKS",
    "SSK": "Saskatchewan Roughriders",
    "TOR": "Toronto Argonauts",  "WPG": "Winnipeg Blue Bombers",
}
TEAM_LOGOS = {
    "BC":  "assets/logos/BC.png",  "CGY": "assets/logos/cal.png",
    "EDM": "assets/logos/edm.png", "HAM": "assets/logos/Ham.png",
    "MTL": "assets/logos/mtl.png", "OTT": "assets/logos/Ott.png",
    "SSK": "assets/logos/ssk.png", "TOR": "assets/logos/Tor.png",
    "WPG": "assets/logos/wpg.png",
}

def get_logo_path(team_code):
    p = BASE / TEAM_LOGOS.get(team_code, "")
    return p if p.exists() else None

@st.cache_resource
def load_model():
    data_path = BASE / "CFL PLAY BY PLAY.xlsx"
    df = pd.read_excel(data_path)
    df = df[df["play_type"].isin(["Pass", "Run", "Sack"])].copy()
    df["called_pass"] = np.where(df["play_type"].isin(["Pass", "Sack"]), 1, 0)
    df["away_team"] = df["game"].str.extract(r"- (\w+) @")
    df["home_team"] = df["game"].str.extract(r"@ (\w+)")
    df["score_diff_offense"] = np.where(
        df["possession_team"] == df["home_team"],
        df["home_team_score"] - df["away_team_score"],
        df["away_team_score"] - df["home_team_score"],
    )
    df["yards_to_go"]               = df["yards_to_go"].fillna(df["yards_to_go"].median())
    df["yards_to_endzone"]          = df["yards_to_endzone"].fillna(df["yards_to_endzone"].median())
    df["seconds_in_half_remaining"] = df["seconds_in_half_remaining"].fillna(df["seconds_in_half_remaining"].median())
    df["defensive_team"]            = df["defensive_team"].fillna("UNK")
    df["down_x_yards"]              = df["down"] * df["yards_to_go"]
    df["is_2nd_short"]              = ((df["down"] == 2) & (df["yards_to_go"] <= 3)).astype(int)
    df["garbage_time"]              = ((df["score_diff_offense"].abs() > 20) & (df["seconds_in_half_remaining"] < 600)).astype(int)
    df["leading_late"]              = ((df["score_diff_offense"] > 7) & (df["seconds_in_half_remaining"] < 300)).astype(int)
    df["two_min_trailing"]          = ((df["seconds_in_half_remaining"] <= 120) & (df["score_diff_offense"] < 0)).astype(int)
    le_off = LabelEncoder()
    le_def = LabelEncoder()
    df["possession_team_enc"] = le_off.fit_transform(df["possession_team"])
    df["defensive_team_enc"]  = le_def.fit_transform(df["defensive_team"])
    features = [
        "possession_team_enc", "defensive_team_enc",
        "quarter", "down", "yards_to_go", "yards_to_endzone",
        "seconds_in_half_remaining", "score_diff_offense",
        "down_x_yards", "is_2nd_short", "garbage_time",
        "leading_late", "two_min_trailing",
    ]
    df_clean = df.dropna(subset=features + ["called_pass"])
    model = HistGradientBoostingClassifier(
        max_iter=500, max_depth=6, learning_rate=0.04,
        min_samples_leaf=12, l2_regularization=0.5,
        class_weight="balanced", random_state=42,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=25,
    )
    model.fit(df_clean[features], df_clean["called_pass"])
    return {
        "model": model, "le_offense": le_off, "le_defense": le_def,
        "features": features, "offense_teams": le_off.classes_.tolist(),
        "defense_teams": le_def.classes_.tolist(),
    }

@st.cache_data
def load_team_lookup():
    return pd.read_csv(BASE / "cfl_team_bucket_lookup.csv")

@st.cache_data
def load_comparables():
    return pd.read_csv(BASE / "cfl_historical_comparables_v2.csv")

@st.cache_data
def load_metrics():
    with open(BASE / "cfl_model_metrics_v2.json", "r") as f:
        return json.load(f)

def get_bucket_static(lookup, team, down, ytg, yte, score_diff, sec):
    sub = lookup[
        (lookup["possession_team"] == team) &
        (lookup["down"] == down) &
        (lookup["distance_bucket"] == distance_bucket(ytg)) &
        (lookup["field_bucket"] == field_bucket(yte)) &
        (lookup["score_bucket"] == score_bucket(score_diff)) &
        (lookup["time_bucket"] == time_bucket(sec))
    ]
    return None if sub.empty else sub.sort_values("plays", ascending=False).iloc[0].to_dict()

@st.cache_data
def compute_tendencies(_team_lookup):
    rows = []
    teams = sorted(_team_lookup["possession_team"].dropna().unique().tolist())

    for team in teams:
        for q in [1, 2, 3, 4]:
            for d in [1, 2]:
                for mins in [12, 9, 6, 3]:
                    for ytg in [2.0, 5.0, 10.0]:
                        for side in ["Own", "Opp"]:
                            for ball in [15.0, 30.0, 45.0]:
                                for score in [sb = score_bucket(float(score))]:
                                    try:
                                        yte = yte_calc(side, ball)
                                        sec = half_seconds(q, mins, 0)
                                        db  = distance_bucket(ytg)
                                        fb  = field_bucket(yte)
                                        sb  = score_bucket(score)
                                        tb  = time_bucket(sec)
                                        sub = _team_lookup[
                                            (_team_lookup["possession_team"] == team) &
                                            (_team_lookup["down"] == d) &
                                            (_team_lookup["distance_bucket"] == db) &
                                            (_team_lookup["field_bucket"] == fb) &
                                            (_team_lookup["score_bucket"] == sb) &
                                            (_team_lookup["time_bucket"] == tb)
                                        ]
                                        if sub.empty:
                                            continue
                                        lk = sub.sort_values("plays", ascending=False).iloc[0]
                                        delta = float(lk["pass_prob_delta_vs_league"])
                                        if abs(delta) >= 0.20 and int(lk["plays"]) >= 10 and int(lk["league_plays"]) >= 20:
                                            rows.append({
                                                "team":               team,
                                                "quarter":            q,
                                                "down":               d,
                                                "minutes":            mins,
                                                "seconds":            0,
                                                "yards_to_go":        ytg,
                                                "field_side":         side,
                                                "ball_on":            ball,
                                                "score_diff":         score,
                                                "delta_vs_league":    delta,
                                                "abs_delta":          abs(delta),
                                                "team_hist_pass_rate": float(lk["pass_prob_hist"]),
                                                "league_pass_rate":   float(lk["league_pass_prob_hist"]),
                                                "tendency":           "Pass-heavy" if delta > 0 else "Run-heavy",
                                                "team_plays":         int(lk["plays"]),
                                                "league_plays":       int(lk["league_plays"]),
                                            })
                                    except Exception:
                                        continue

    if not rows:
        return pd.DataFrame(columns=["team","quarter","down","minutes","seconds",
                                     "yards_to_go","field_side","ball_on","score_diff",
                                     "delta_vs_league","abs_delta","team_hist_pass_rate",
                                     "league_pass_rate","tendency","team_plays","league_plays"])
    return pd.DataFrame(rows).sort_values(["team","abs_delta"], ascending=[True,False]).reset_index(drop=True)

MODEL_PKG   = load_model()
MODEL       = MODEL_PKG["model"]
LE_OFF      = MODEL_PKG["le_offense"]
LE_DEF      = MODEL_PKG["le_defense"]
FEATURES    = MODEL_PKG["features"]
TEAM_LOOKUP = load_team_lookup()
COMPS       = load_comparables()
METRICS     = load_metrics()
TENDENCIES  = compute_tendencies(TEAM_LOOKUP)
st.write("Tendencies shape:", TENDENCIES.shape)
st.write("Columns:", TENDENCIES.columns.tolist())
st.write("Bucket sample:", TEAM_LOOKUP["field_bucket"].unique())
TEAMS       = sorted(TEAM_LOOKUP["possession_team"].dropna().astype(str).unique().tolist())


def distance_bucket(x):
    return "Short (1-3)" if x <= 3 else ("Medium (4-7)" if x <= 7 else "Long (8+)")

def field_bucket(x):
    if x <= 20: return "Red Zone"
    if x <= 40: return "Opponent Territory"
    if x <= 60: return "Midfield"
    if x <= 80: return "Own Territory"
    return "Own Deep"

def score_bucket(x):
    x = float(x)
    if x <= -8: return "Trailing 8+"
    if x <= -1: return "Trailing 1-7"
    if x == 0:  return "Tied"
    if x <= 7:  return "Leading 1-7"
    return "Leading 8+"

def time_bucket(s):
    return "2-min" if s <= 120 else ("Late Half" if s <= 600 else "Early Half")

def half_seconds(quarter, mins, secs):
    rem = int(mins) * 60 + int(secs)
    return rem + 900 if int(quarter) in (1, 3) else rem

def yte_calc(side, yardline):
    y = float(yardline)
    return 110 - y if side == "Own" else y

def encode_team(le, team):
    return int(le.transform([team])[0]) if team in le.classes_ else 0

def build_row(team, def_team, quarter, mins, secs, down, ytg, side, ball_on, score_diff):
    yte = yte_calc(side, ball_on)
    sec = half_seconds(quarter, mins, secs)
    row = {
        "possession_team_enc":       encode_team(LE_OFF, team),
        "defensive_team_enc":        encode_team(LE_DEF, def_team),
        "quarter":                   quarter,
        "down":                      down,
        "yards_to_go":               ytg,
        "yards_to_endzone":          yte,
        "seconds_in_half_remaining": sec,
        "score_diff_offense":        score_diff,
        "down_x_yards":              down * ytg,
        "is_2nd_short":              int(down == 2 and ytg <= 3),
        "garbage_time":              int(abs(score_diff) > 20 and sec < 600),
        "leading_late":              int(score_diff > 7 and sec < 300),
        "two_min_trailing":          int(sec <= 120 and score_diff < 0),
    }
    return pd.DataFrame([row])[FEATURES], yte, sec

def get_bucket(team, down, ytg, yte, score_diff, sec):
    sub = TEAM_LOOKUP[
        (TEAM_LOOKUP["possession_team"] == team) &
        (TEAM_LOOKUP["down"] == down) &
        (TEAM_LOOKUP["distance_bucket"] == distance_bucket(ytg)) &
        (TEAM_LOOKUP["field_bucket"] == field_bucket(yte)) &
        (TEAM_LOOKUP["score_bucket"] == score_bucket(score_diff)) &
        (TEAM_LOOKUP["time_bucket"] == time_bucket(sec))
    ]
    return None if sub.empty else sub.sort_values("plays", ascending=False).iloc[0].to_dict()

def get_comparables(team, quarter, down, ytg, yte, sec, score_diff, n=10):
    d = COMPS[(COMPS["possession_team"] == team) & (COMPS["down"] == down)].copy()
    if d.empty:
        return d
    d["dist"] = (
        ((d["quarter"] - quarter) / 2.0) ** 2 +
        ((d["yards_to_go"] - ytg) / 4.0) ** 2 +
        ((d["yards_to_endzone"] - yte) / 15.0) ** 2 +
        ((d["seconds_in_half_remaining"] - sec) / 240.0) ** 2 +
        ((d["score_diff_offense"] - score_diff) / 7.0) ** 2
    )
    d = d.sort_values("dist").head(n).copy()
    d["Play"] = np.where(d["called_pass"] == 1, "PASS", "RUN")
    return d[["Play", "play_result", "description", "cfl_game_id", "quarter",
              "yards_to_go", "score_diff_offense"]]


# ── SESSION DEFAULTS ──────────────────────────────────────────────────────────
for k, v in dict(team="WPG", def_team="MTL", quarter=1, minutes=4, seconds=0,
                 down=1, ytg=10.0, field_side="Own", ball_on=30.0, score_diff=-10).items():
    st.session_state.setdefault(k, v)

team   = st.session_state["team"]
colors = TEAM_COLORS.get(team, {"primary": "#1a1a2e", "secondary": "#e94560"})
pri    = colors["primary"]

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;800&family=Barlow:wght@400;500;600&display=swap');

*, *::before, *::after {{ box-sizing: border-box; }}
html, body, .stApp {{ background-color: #0a0e1a !important; color: #e8eaf0 !important; font-family: 'Barlow', sans-serif !important; }}
.block-container {{ padding: 0 !important; max-width: 100% !important; }}

.scout-header {{ background: linear-gradient(135deg, #0d1117 0%, #0a0e1a 60%, {pri}22 100%); border-bottom: 2px solid {pri}; padding: 0; display: flex; align-items: stretch; min-height: 72px; }}
.scout-header-left {{ padding: 14px 28px; display: flex; flex-direction: column; justify-content: center; border-right: 1px solid #1e2433; min-width: 300px; }}
.scout-wordmark {{ font-family: 'Barlow Condensed', sans-serif; font-size: 1.05rem; font-weight: 600; letter-spacing: 0.25em; text-transform: uppercase; color: {pri}; line-height: 1; margin-bottom: 3px; }}
.scout-title {{ font-family: 'Barlow Condensed', sans-serif; font-size: 1.75rem; font-weight: 800; color: #ffffff; line-height: 1; letter-spacing: 0.02em; }}
.scout-header-right {{ flex: 1; display: flex; align-items: center; justify-content: flex-end; padding: 14px 28px; gap: 20px; }}
.team-badge {{ display: flex; align-items: center; gap: 12px; background: {pri}18; border: 1px solid {pri}44; border-radius: 8px; padding: 8px 16px; }}
.team-badge-name {{ font-family: 'Barlow Condensed', sans-serif; font-size: 1.3rem; font-weight: 700; color: #ffffff; line-height: 1; }}
.team-badge-label {{ font-size: 0.7rem; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; color: {pri}; line-height: 1; margin-top: 3px; }}

.live-bar {{ background: #0d1117; border-bottom: 1px solid #1e2433; padding: 10px 28px; display: flex; align-items: center; gap: 28px; flex-wrap: wrap; }}
.live-tag {{ background: {pri}; color: white; font-family: 'Barlow Condensed', sans-serif; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.15em; padding: 3px 8px; border-radius: 3px; }}
.live-item {{ display: flex; flex-direction: column; gap: 1px; }}
.live-label {{ font-size: 0.62rem; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; color: #94a3b8; line-height: 1; }}
.live-value {{ font-family: 'Barlow Condensed', sans-serif; font-size: 1.05rem; font-weight: 700; color: #e8eaf0; line-height: 1; }}
.live-sep {{ width: 1px; height: 28px; background: #1e2433; }}

.prob-hero {{ background: linear-gradient(135deg, #0d1117 0%, #111827 100%); border-bottom: 1px solid #1e2433; padding: 24px 28px; display: flex; gap: 16px; align-items: stretch; }}
.prob-card {{ flex: 1; border-radius: 10px; padding: 18px 20px; display: flex; flex-direction: column; gap: 4px; }}
.prob-card-pass {{ background: {pri}20; border: 1.5px solid {pri}66; }}
.prob-card-run {{ background: #1a2235; border: 1.5px solid #2a3448; }}
.prob-card-stat {{ background: #0d1117; border: 1.5px solid #1e2433; flex: 0.6; }}
.prob-card-label {{ font-size: 0.68rem; font-weight: 700; letter-spacing: 0.15em; text-transform: uppercase; color: #94a3b8; line-height: 1; }}
.prob-card-value {{ font-family: 'Barlow Condensed', sans-serif; font-size: 3.2rem; font-weight: 800; line-height: 1; }}
.prob-card-pass .prob-card-value {{ color: {pri}; }}
.prob-card-run .prob-card-value {{ color: #94a3b8; }}
.prob-card-stat .prob-card-value {{ font-size: 2rem; color: #e8eaf0; }}
.prob-bar-track {{ height: 4px; background: #1e2433; border-radius: 2px; margin-top: 8px; overflow: hidden; }}
.prob-bar-fill {{ height: 100%; border-radius: 2px; }}

.section-title {{ font-family: 'Barlow Condensed', sans-serif; font-size: 1rem; font-weight: 800; letter-spacing: 0.2em; text-transform: uppercase; color: #7bafd4; margin: 0 0 12px 0; padding-bottom: 6px; border-bottom: 1px solid {pri}33; }}
.comp-row {{ display: flex; gap: 12px; margin-top: 12px; }}
.comp-stat {{ flex: 1; background: #0d1117; border: 1px solid #1e2433; border-radius: 8px; padding: 12px 14px; }}
.comp-stat-label {{ font-size: 0.62rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: #94a3b8; margin-bottom: 4px; }}
.comp-stat-value {{ font-family: 'Barlow Condensed', sans-serif; font-size: 1.5rem; font-weight: 700; color: #e2e8f0; line-height: 1; }}
.comp-stat-value.positive {{ color: #22c55e; }}
.comp-stat-value.negative {{ color: #ef4444; }}
.comp-stat-value.neutral  {{ color: {pri}; }}

.alert-box {{ border-radius: 8px; padding: 12px 16px; margin-top: 12px; font-size: 0.85rem; line-height: 1.4; }}
.alert-pass    {{ background: {pri}15; border-left: 3px solid {pri}; color: #cbd5e1; }}
.alert-run     {{ background: #ef444415; border-left: 3px solid #ef4444; color: #cbd5e1; }}
.alert-neutral {{ background: #111827; border-left: 3px solid #374151; color: #94a3b8; }}

section[data-testid="stSidebar"] {{ background: #0d1117 !important; border-right: 1px solid #1e2433 !important; }}
section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p {{ color: #94a3b8 !important; font-size: 0.78rem !important; font-weight: 600 !important; letter-spacing: 0.08em !important; text-transform: uppercase !important; }}
section[data-testid="stSidebar"] .stSelectbox > div > div, section[data-testid="stSidebar"] input {{ background: #111827 !important; border: 1px solid #1e2433 !important; color: #e2e8f0 !important; border-radius: 6px !important; }}
.sidebar-heading {{ font-family: 'Barlow Condensed', sans-serif; font-size: 0.9rem; font-weight: 800; letter-spacing: 0.2em; text-transform: uppercase; color: #7bafd4; margin: 16px 0 8px 0; padding-bottom: 4px; border-bottom: 1px solid {pri}33; }}
.stTabs [data-baseweb="tab-list"] {{ background: #0d1117 !important; border-bottom: 1px solid #1e2433 !important; padding: 0 28px !important; gap: 0 !important; }}
.stTabs [data-baseweb="tab"] {{ font-family: 'Barlow Condensed', sans-serif !important; font-size: 0.8rem !important; font-weight: 700 !important; letter-spacing: 0.12em !important; text-transform: uppercase !important; color: #94a3b8 !important; padding: 12px 20px !important; background: transparent !important; }}
.stTabs [aria-selected="true"] {{ color: {pri} !important; border-bottom: 2px solid {pri} !important; }}
.stTabs [data-baseweb="tab-panel"] {{ background: #0a0e1a !important; padding: 22px 28px !important; }}

#MainMenu, footer, header {{ visibility: hidden; }}
.stDeployButton {{ display: none; }}
div[data-testid="stToolbar"] {{ display: none; }}
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f'<div class="sidebar-heading">Offense</div>', unsafe_allow_html=True)
    st.selectbox("Team", TEAMS, key="team", label_visibility="collapsed")
    st.markdown(f'<div class="sidebar-heading">Defense</div>', unsafe_allow_html=True)
    st.selectbox("Defense", TEAMS, key="def_team", label_visibility="collapsed")
    st.markdown('<div class="sidebar-heading">Game Clock</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1: st.selectbox("Qtr", [1,2,3,4], key="quarter")
    with c2: st.selectbox("Down", [1,2,3], key="down")
    c3, c4 = st.columns(2)
    with c3: st.number_input("Min", 0, 15, step=1, key="minutes")
    with c4: st.number_input("Sec", 0, 59, step=1, key="seconds")
    st.markdown('<div class="sidebar-heading">Field Position</div>', unsafe_allow_html=True)
    st.selectbox("Side", ["Own", "Opp"], key="field_side")
    st.number_input("Yard line", 1.0, 55.0, step=1.0, key="ball_on")
    st.number_input("Yards to go", 0.0, 50.0, step=1.0, key="ytg")
    st.markdown('<div class="sidebar-heading">Score</div>', unsafe_allow_html=True)
    st.number_input("Score diff (offense)", -60, 60, step=1, key="score_diff")
    st.caption("Negative = trailing")
    if st.session_state["down"] == 3:
        st.warning("3rd down is typically a kicking situation in the CFL.")

# ── READ INPUTS ───────────────────────────────────────────────────────────────
team       = st.session_state["team"]
def_team   = st.session_state["def_team"]
quarter    = st.session_state["quarter"]
minutes    = st.session_state["minutes"]
seconds    = st.session_state["seconds"]
down       = st.session_state["down"]
ytg        = float(st.session_state["ytg"])
field_side = st.session_state["field_side"]
ball_on    = float(st.session_state["ball_on"])
score_diff = int(st.session_state["score_diff"])

x, yte, sec_half = build_row(team, def_team, quarter, minutes, seconds, down, ytg, field_side, ball_on, score_diff)
pass_prob = float(MODEL.predict_proba(x)[0, 1])
run_prob  = 1 - pass_prob
lookup    = get_bucket(team, down, ytg, yte, score_diff, sec_half)
comps     = get_comparables(team, quarter, down, ytg, yte, sec_half, score_diff)
logo_path = get_logo_path(team)

# ── HEADER ────────────────────────────────────────────────────────────────────
logo_html = ""
if logo_path:
    with open(str(logo_path), "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    logo_html = '<img src="data:image/png;base64,' + b64 + '" height="42" style="opacity:0.95">'

st.markdown(f"""
<div class="scout-header">
  <div class="scout-header-left">
    <div class="scout-wordmark">CFL Analytics</div>
    <div class="scout-title">Play Tendency Scout</div>
  </div>
  <div class="scout-header-right">
    <div class="team-badge">
      {logo_html}
      <div>
        <div class="team-badge-name">{TEAM_NAMES.get(team, team)}</div>
        <div class="team-badge-label">Offense · vs {def_team}</div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── LIVE BAR ──────────────────────────────────────────────────────────────────
down_str = {1: "1st", 2: "2nd", 3: "3rd"}.get(down, f"{down}th")
st.markdown(f"""
<div class="live-bar">
  <span class="live-tag">LIVE</span>
  <div class="live-item"><div class="live-label">Situation</div><div class="live-value">{down_str} &amp; {ytg:g}</div></div>
  <div class="live-sep"></div>
  <div class="live-item"><div class="live-label">Field</div><div class="live-value">{field_side} {ball_on:g}</div></div>
  <div class="live-sep"></div>
  <div class="live-item"><div class="live-label">Clock</div><div class="live-value">Q{quarter} · {minutes}:{str(seconds).zfill(2)}</div></div>
  <div class="live-sep"></div>
  <div class="live-item"><div class="live-label">Score diff</div><div class="live-value">{score_diff:+d}</div></div>
  <div class="live-sep"></div>
  <div class="live-item"><div class="live-label">Distance</div><div class="live-value">{distance_bucket(ytg)}</div></div>
  <div class="live-sep"></div>
  <div class="live-item"><div class="live-label">Field zone</div><div class="live-value">{field_bucket(yte)}</div></div>
  <div class="live-sep"></div>
  <div class="live-item"><div class="live-label">Game state</div><div class="live-value">{score_bucket(score_diff)} · {time_bucket(sec_half)}</div></div>
</div>
""", unsafe_allow_html=True)

# ── PROBABILITY HERO ──────────────────────────────────────────────────────────
acc = METRICS.get("accuracy_at_0_5_threshold", 0)
auc = METRICS.get("roc_auc", 0)
n   = int(lookup["plays"]) if lookup else 0
st.markdown(f"""
<div class="prob-hero">
  <div class="prob-card prob-card-pass">
    <div class="prob-card-label">Pass probability</div>
    <div class="prob-card-value">{pass_prob:.1%}</div>
    <div class="prob-bar-track"><div class="prob-bar-fill" style="width:{pass_prob*100:.1f}%;background:{pri}"></div></div>
  </div>
  <div class="prob-card prob-card-run">
    <div class="prob-card-label">Run probability</div>
    <div class="prob-card-value">{run_prob:.1%}</div>
    <div class="prob-bar-track"><div class="prob-bar-fill" style="width:{run_prob*100:.1f}%;background:#475569"></div></div>
  </div>
  <div class="prob-card prob-card-stat">
    <div class="prob-card-label">Model accuracy</div>
    <div class="prob-card-value">{acc:.1%}</div>
  </div>
  <div class="prob-card prob-card-stat">
    <div class="prob-card-label">ROC-AUC</div>
    <div class="prob-card-value">{auc:.3f}</div>
  </div>
  <div class="prob-card prob-card-stat">
    <div class="prob-card-label">Bucket sample</div>
    <div class="prob-card-value">{n}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── CONTENT ───────────────────────────────────────────────────────────────────
left, right = st.columns(2)

with left:
    st.markdown('<div class="section-title">Team tendency vs league</div>', unsafe_allow_html=True)
    if lookup:
        team_hist  = float(lookup["pass_prob_hist"])
        league_avg = float(lookup["league_pass_prob_hist"])
        delta      = float(lookup["pass_prob_delta_vs_league"])
        n_plays    = int(lookup["plays"])
        n_league   = int(lookup["league_plays"])
        delta_cls  = "positive" if delta >= 0.05 else ("negative" if delta <= -0.05 else "neutral")
        delta_sign = "▲" if delta > 0 else ("▼" if delta < 0 else "—")
        st.markdown(f"""
<div class="comp-row">
  <div class="comp-stat"><div class="comp-stat-label">Model prediction</div><div class="comp-stat-value neutral">{pass_prob:.1%}</div></div>
  <div class="comp-stat"><div class="comp-stat-label">{team} hist. pass rate</div><div class="comp-stat-value">{team_hist:.1%}</div></div>
  <div class="comp-stat"><div class="comp-stat-label">League avg</div><div class="comp-stat-value">{league_avg:.1%}</div></div>
  <div class="comp-stat"><div class="comp-stat-label">Team vs league</div><div class="comp-stat-value {delta_cls}">{delta_sign} {abs(delta):.1%}</div></div>
</div>
<div style="margin-top:8px;font-size:0.75rem;color:#94a3b8;">{n_plays} team plays · {n_league} league plays in this bucket</div>
""", unsafe_allow_html=True)
        if delta >= 0.20:
            st.markdown(f'<div class="alert-box alert-pass"><strong>Pass-heavy tendency.</strong> {team} passes {delta:+.1%} above league average here.</div>', unsafe_allow_html=True)
        elif delta <= -0.20:
            st.markdown(f'<div class="alert-box alert-run"><strong>Run-heavy tendency.</strong> {team} passes {delta:+.1%} vs league average — expect the run.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-box alert-neutral">Within ±20% of league average. No strong tendency signal.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert-box alert-neutral">No exact bucket match for this situation.</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-title">Closest historical plays</div>', unsafe_allow_html=True)
    if comps.empty:
        st.markdown('<div class="alert-box alert-neutral">No comparable plays found.</div>', unsafe_allow_html=True)
    else:
        rows_html = ""
        for _, row in comps.iterrows():
            play_color = pri if row["Play"] == "PASS" else "#475569"
            desc = str(row["description"])[:72] + "…" if len(str(row["description"])) > 72 else str(row["description"])
            rows_html += f"""
            <div class="play-row">
              <div class="play-badge" style="background:{play_color}22;border-color:{play_color}66;color:{play_color}">{row["Play"]}</div>
              <div class="play-meta">
                <div class="play-desc">{desc}</div>
                <div class="play-tags">
                  <span class="play-tag">Q{int(row["quarter"])}</span>
                  <span class="play-tag">{int(row["yards_to_go"])} yds to go</span>
                  <span class="play-tag">{int(row["score_diff_offense"]):+d} score</span>
                  <span class="play-tag">{row["play_result"]}</span>
                </div>
              </div>
            </div>"""

        st.markdown(f"""
<style>
.play-feed {{ display: flex; flex-direction: column; gap: 6px; margin-top: 4px; }}
.play-row {{ display: flex; align-items: flex-start; gap: 12px; background: #0d1117; border: 1px solid #1e2433; border-radius: 8px; padding: 10px 14px; transition: border-color 0.15s; }}
.play-row:hover {{ border-color: {pri}44; }}
.play-badge {{ font-family: 'Barlow Condensed', sans-serif; font-size: 0.72rem; font-weight: 800; letter-spacing: 0.12em; border: 1px solid; border-radius: 4px; padding: 3px 8px; min-width: 48px; text-align: center; margin-top: 2px; flex-shrink: 0; }}
.play-meta {{ flex: 1; min-width: 0; }}
.play-desc {{ font-size: 0.82rem; color: #cbd5e1; line-height: 1.35; margin-bottom: 5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.play-tags {{ display: flex; gap: 6px; flex-wrap: wrap; }}
.play-tag {{ font-size: 0.65rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #64748b; background: #111827; border: 1px solid #1e2433; border-radius: 3px; padding: 2px 6px; }}
</style>
<div class="play-feed">{rows_html}</div>
""", unsafe_allow_html=True)


# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, = st.tabs(["TOP TENDENCIES"])

with tab1:
    st.markdown(f'<div class="section-title">Top 3 outlier tendencies — {TEAM_NAMES.get(team, team)}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:0.78rem;color:#64748b;margin-bottom:16px;">Situations where {team} deviates ≥20% from league average · min. 10 team plays · min. 20 league plays</div>', unsafe_allow_html=True)

    if TENDENCIES is not None:
        team_tends = TENDENCIES[TENDENCIES["team"] == team].copy()
        if "abs_delta" not in team_tends.columns:
            team_tends["abs_delta"] = team_tends["delta_vs_league"].abs()
        team_tends = team_tends.sort_values("abs_delta", ascending=False).head(3)

        if team_tends.empty:
            st.markdown('<div class="alert-box alert-neutral">No strong tendencies found for this team.</div>', unsafe_allow_html=True)
        else:
            for _, row in team_tends.iterrows():
                delta      = float(row["delta_vs_league"])
                league_rate= float(row["league_pass_rate"])
                team_rate  = float(row.get("team_hist_pass_rate", league_rate + delta))
                is_pass    = delta > 0
                card_color = pri if is_pass else "#ef4444"
                tend_label = "PASS HEAVY" if is_pass else "RUN HEAVY"
                bar_pct    = min(abs(delta) * 100 / 60, 100)

                # Situation labels
                down_str  = {1:"1st", 2:"2nd", 3:"3rd"}.get(int(row["down"]), "—")
                ytg_str   = f"{int(row['yards_to_go'])} yds"
                side_str  = f"{row['field_side']} {int(row['ball_on'])}"
                score_str = f"{int(row['score_diff']):+d}"
                clock_str = f"Q{int(row['quarter'])} {int(row['minutes'])}:{int(row['seconds']):02d}"

                # Pull comparable plays for this scenario
                sc_yte = yte_calc(row["field_side"], row["ball_on"])
                sc_sec = half_seconds(int(row["quarter"]), int(row["minutes"]), int(row["seconds"]))
                sc_comps = get_comparables(
                    team, int(row["quarter"]), int(row["down"]),
                    float(row["yards_to_go"]), sc_yte, sc_sec,
                    int(row["score_diff"]), n=3
                )

                comp_rows_html = ""
                if not sc_comps.empty:
                    for _, cr in sc_comps.iterrows():
                        cp_color = pri if cr["Play"] == "PASS" else "#475569"
                        desc = str(cr["description"])[:65] + "…" if len(str(cr["description"])) > 65 else str(cr["description"])
                        comp_rows_html += f"""
                        <div class="tend-comp-row">
                          <div class="play-badge" style="background:{cp_color}22;border-color:{cp_color}66;color:{cp_color};font-size:0.6rem;padding:2px 6px;">{cr["Play"]}</div>
                          <div style="font-size:0.76rem;color:#94a3b8;flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{desc}</div>
                        </div>"""

                st.markdown(f"""
<style>
.tend-card {{
    background: #0d1117;
    border: 1px solid #1e2433;
    border-left: 3px solid {card_color};
    border-radius: 10px;
    padding: 18px 20px;
    margin-bottom: 12px;
}}
.tend-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 14px;
}}
.tend-badge {{
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.14em;
    background: {card_color}22;
    border: 1px solid {card_color}66;
    color: {card_color};
    border-radius: 4px;
    padding: 3px 10px;
}}
.tend-delta {{
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    color: {card_color};
    line-height: 1;
}}
.tend-sit-grid {{
    display: flex;
    gap: 8px;
    margin-bottom: 14px;
    flex-wrap: wrap;
}}
.tend-sit-pill {{
    background: #111827;
    border: 1px solid #1e2433;
    border-radius: 5px;
    padding: 5px 10px;
}}
.tend-sit-label {{
    font-size: 0.58rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4a5568;
    margin-bottom: 2px;
}}
.tend-sit-value {{
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.95rem;
    font-weight: 700;
    color: #e2e8f0;
    line-height: 1;
}}
.tend-bar-track {{
    height: 6px;
    background: #1e2433;
    border-radius: 3px;
    margin-bottom: 6px;
    overflow: hidden;
}}
.tend-bar-fill {{
    height: 100%;
    background: {card_color};
    border-radius: 3px;
    width: {bar_pct:.1f}%;
}}
.tend-bar-labels {{
    display: flex;
    justify-content: space-between;
    font-size: 0.68rem;
    color: #4a5568;
    margin-bottom: 14px;
}}
.tend-comps-title {{
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #4a5568;
    margin-bottom: 8px;
}}
.tend-comp-row {{
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 0;
    border-bottom: 1px solid #1e2433;
}}
.tend-comp-row:last-child {{ border-bottom: none; }}
</style>
<div class="tend-card">
  <div class="tend-header">
    <div class="tend-badge">{tend_label}</div>
    <div class="tend-delta">{delta:+.1%} vs league</div>
  </div>
  <div class="tend-sit-grid">
    <div class="tend-sit-pill"><div class="tend-sit-label">Down</div><div class="tend-sit-value">{down_str} &amp; {ytg_str}</div></div>
    <div class="tend-sit-pill"><div class="tend-sit-label">Field</div><div class="tend-sit-value">{side_str}</div></div>
    <div class="tend-sit-pill"><div class="tend-sit-label">Clock</div><div class="tend-sit-value">{clock_str}</div></div>
    <div class="tend-sit-pill"><div class="tend-sit-label">Score diff</div><div class="tend-sit-value">{score_str}</div></div>
    <div class="tend-sit-pill"><div class="tend-sit-label">{team} pass rate</div><div class="tend-sit-value">{team_rate:.1%}</div></div>
    <div class="tend-sit-pill"><div class="tend-sit-label">League avg</div><div class="tend-sit-value">{league_rate:.1%}</div></div>
  </div>
  <div class="tend-bar-track"><div class="tend-bar-fill"></div></div>
  <div class="tend-bar-labels">
    <span>0%</span><span>Deviation from league avg</span><span>60%+</span>
  </div>
  <div class="tend-comps-title">Example plays from this situation</div>
  <div>{comp_rows_html if comp_rows_html else '<div style="font-size:0.76rem;color:#4a5568;">No comparable plays found.</div>'}</div>
</div>
""", unsafe_allow_html=True)
    else:
        st.info("Run `precompute_tendencies.py` and upload `cfl_top5_tendencies_precomputed.csv` to enable this tab.")
