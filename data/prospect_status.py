import pandas as pd


HITTER_GRADUATION_PA = 130
PITCHER_GRADUATION_IP = 50


def prospect_graduation_playing_time(row):
    player_type = str(row.get("Player_Type", "") or "").strip().lower()
    positions = str(row.get("Positions", row.get("Display_POS", "")) or "").upper()
    if player_type == "pitcher" or any(pos in positions.split("/") for pos in {"SP", "RP", "P"}):
        return pd.to_numeric(row.get("YTD_IP"), errors="coerce"), PITCHER_GRADUATION_IP
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
    out["Prospect_Listed"] = out["Is_Prospect"].fillna(False).astype(bool)
    out["Is_Prospect"] = out.apply(is_ungraduated_prospect, axis=1)
    return out
