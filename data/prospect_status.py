import os

import pandas as pd


HITTER_GRADUATION_PA = 130
PITCHER_GRADUATION_IP = 50
PITCHER_GRADUATION_OUTS = PITCHER_GRADUATION_IP * 3

DEFAULT_HITTER_PLAYING_TIME_FILES = [
    "2025_performance_hitters.csv",
    "ytd_fgpts_hitters.csv",
]
DEFAULT_PITCHER_PLAYING_TIME_FILES = [
    "2025_performance_pitchers.csv",
    "ytd_fgpts_pitchers.csv",
]


def normalized_player_id(value):
    if pd.isna(value):
        return None
    try:
        return str(int(float(value)))
    except ValueError:
        clean = str(value).strip()
        return clean or None


def baseball_ip_to_outs(value):
    if pd.isna(value):
        return 0
    text = str(value).strip()
    if not text:
        return 0
    whole_text, _, frac_text = text.partition(".")
    try:
        whole = int(float(whole_text or 0))
    except ValueError:
        return 0
    if not frac_text:
        return whole * 3
    frac_digit = int(frac_text[:1] or 0)
    if frac_digit not in {0, 1, 2}:
        frac_digit = 0
    return whole * 3 + frac_digit


def outs_to_baseball_ip(outs):
    outs = int(outs or 0)
    whole, partial = divmod(outs, 3)
    return float(f"{whole}.{partial}") if partial else float(whole)


def load_recent_mlb_playing_time(
    hitter_paths=None,
    pitcher_paths=None,
):
    hitter_paths = hitter_paths or DEFAULT_HITTER_PLAYING_TIME_FILES
    pitcher_paths = pitcher_paths or DEFAULT_PITCHER_PLAYING_TIME_FILES
    rows = []
    for path in hitter_paths:
        if not path or not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if "PlayerId" not in df.columns or "PA" not in df.columns:
            continue
        temp = pd.DataFrame({
            "PlayerIdKey": df["PlayerId"].apply(normalized_player_id),
            "Career_MLB_PA": pd.to_numeric(df["PA"], errors="coerce").fillna(0),
            "Career_MLB_Outs": 0,
        })
        rows.append(temp)
    for path in pitcher_paths:
        if not path or not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if "PlayerId" not in df.columns or "IP" not in df.columns:
            continue
        temp = pd.DataFrame({
            "PlayerIdKey": df["PlayerId"].apply(normalized_player_id),
            "Career_MLB_PA": 0,
            "Career_MLB_Outs": df["IP"].apply(baseball_ip_to_outs),
        })
        rows.append(temp)
    if not rows:
        return pd.DataFrame(columns=["PlayerIdKey", "Career_MLB_PA", "Career_MLB_IP", "Career_MLB_Outs"])
    playing_time = pd.concat(rows, ignore_index=True)
    playing_time = playing_time[playing_time["PlayerIdKey"].notna()]
    playing_time = playing_time.groupby("PlayerIdKey", as_index=False).agg(
        Career_MLB_PA=("Career_MLB_PA", "sum"),
        Career_MLB_Outs=("Career_MLB_Outs", "sum"),
    )
    playing_time["Career_MLB_IP"] = playing_time["Career_MLB_Outs"].apply(outs_to_baseball_ip)
    return playing_time[["PlayerIdKey", "Career_MLB_PA", "Career_MLB_IP", "Career_MLB_Outs"]]


def add_recent_mlb_playing_time(df, playing_time=None):
    out = df.copy()
    if "PlayerIdKey" not in out.columns:
        return out
    playing_time = playing_time if playing_time is not None else load_recent_mlb_playing_time()
    if playing_time.empty:
        for col in ["Career_MLB_PA", "Career_MLB_IP", "Career_MLB_Outs"]:
            if col not in out.columns:
                out[col] = pd.NA
        return out
    out = out.merge(playing_time, on="PlayerIdKey", how="left")
    return out


def prospect_graduation_playing_time(row):
    player_type = str(row.get("Player_Type", "") or "").strip().lower()
    positions = str(row.get("Positions", row.get("Display_POS", "")) or "").upper()
    if player_type == "pitcher" or any(pos in positions.split("/") for pos in {"SP", "RP", "P"}):
        career_outs = pd.to_numeric(row.get("Career_MLB_Outs"), errors="coerce")
        if pd.notna(career_outs):
            return float(career_outs), PITCHER_GRADUATION_OUTS
        return baseball_ip_to_outs(row.get("YTD_IP")), PITCHER_GRADUATION_OUTS
    career_pa = pd.to_numeric(row.get("Career_MLB_PA"), errors="coerce")
    if pd.notna(career_pa):
        return float(career_pa), HITTER_GRADUATION_PA
    return pd.to_numeric(row.get("YTD_PA"), errors="coerce"), HITTER_GRADUATION_PA


def is_ungraduated_prospect(row):
    if not bool(row.get("Is_Prospect", False)):
        return False
    playing_time, threshold = prospect_graduation_playing_time(row)
    if pd.isna(playing_time):
        return True
    return float(playing_time) <= threshold


def apply_prospect_graduation(df):
    out = df.copy()
    if "Is_Prospect" not in out.columns:
        return out
    if "Prospect_Listed" not in out.columns:
        out["Prospect_Listed"] = out["Is_Prospect"].fillna(False).astype(bool)
    else:
        out["Prospect_Listed"] = out["Prospect_Listed"].fillna(False).astype(bool)
    out["Is_Prospect"] = out.apply(is_ungraduated_prospect, axis=1)
    return out
