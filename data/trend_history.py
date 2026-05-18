import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Pandas requires version")
import pandas as pd

from data_sources import DEFAULT_CACHE_DIR
from prospect_updates import normalize_text
from snapshots import PLAYER_TRENDS_FILE


HISTORY_COLUMNS = [
    "Snapshot",
    "Snapshot_Date",
    "Team Name",
    "Name",
    "PlayerIdKey",
    "Positions",
    "Eligible_Position_Ranks",
    "Primary_Position",
    "Position_Rank",
    "Player_Role",
    "Salary",
    "Current_Value",
    "Context_Value",
    "Trend_Trade_Adjustment",
    "Trend_Label",
    "Trend_Sample",
    "Prospect_Pedigree_Label",
    "PS_Best_Score",
    "PS_Best_Year",
    "YTD_Value",
    "YTD_ROS_Gap",
    "Projection_Change",
    "Skill_Score",
    "Sample_Confidence",
    "Role_Change",
    "Stock_Label",
    "Trend_Notes",
]


def snapshot_dirs(cache_root=DEFAULT_CACHE_DIR):
    root = Path(cache_root)
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if path.is_dir())


def snapshot_date(snapshot_name):
    parsed = pd.to_datetime(snapshot_name, format="%Y%m%dT%H%M%SZ", utc=True, errors="coerce")
    if pd.isna(parsed):
        return snapshot_name
    return parsed


def read_snapshot_trends(cache_dir):
    trends_path = Path(cache_dir) / PLAYER_TRENDS_FILE
    if not trends_path.exists():
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    df = pd.read_csv(trends_path)
    df["Snapshot"] = Path(cache_dir).name
    df["Snapshot_Date"] = snapshot_date(Path(cache_dir).name)
    for col in HISTORY_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[HISTORY_COLUMNS]


def build_trend_history(cache_root=DEFAULT_CACHE_DIR):
    frames = [read_snapshot_trends(path) for path in snapshot_dirs(cache_root)]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    history = pd.concat(frames, ignore_index=True, sort=False)
    history["PlayerIdKey"] = history["PlayerIdKey"].astype(str)
    return history.sort_values(["PlayerIdKey", "Snapshot"]).reset_index(drop=True)


def snapshot_stage(snapshot_date, latest_date=None):
    if pd.isna(snapshot_date):
        return "snapshot"
    date = pd.Timestamp(snapshot_date).date()
    if date.year <= 2025:
        return "2025 end"
    if date.month <= 3:
        return "2026 start"
    if latest_date is not None and date == latest_date:
        return "today"
    return "in-season"


def compact_history_milestones(history):
    if history.empty:
        return history
    compact = history.copy()
    compact["Snapshot_Date_Parsed"] = pd.to_datetime(compact["Snapshot_Date"], utc=True, errors="coerce")
    valid_dates = compact["Snapshot_Date_Parsed"].dropna()
    latest_date = valid_dates.dt.date.max() if not valid_dates.empty else None
    compact["Snapshot_Stage"] = compact["Snapshot_Date_Parsed"].apply(lambda value: snapshot_stage(value, latest_date))
    compact = compact[compact["Snapshot_Stage"].isin({"2025 end", "2026 start", "today"})].copy()
    if compact.empty:
        return compact.drop(columns=["Snapshot_Date_Parsed"], errors="ignore")

    stage_order = {"2025 end": 0, "2026 start": 1, "today": 2}
    compact["Snapshot_Stage_Order"] = compact["Snapshot_Stage"].map(stage_order)
    compact = (
        compact.sort_values(["PlayerIdKey", "Snapshot_Stage_Order", "Snapshot_Date_Parsed", "Snapshot"])
        .groupby(["PlayerIdKey", "Snapshot_Stage"], as_index=False, sort=False)
        .tail(1)
        .sort_values(["PlayerIdKey", "Snapshot_Stage_Order"])
        .reset_index(drop=True)
    )
    compact["Snapshot_Label"] = compact["Snapshot_Stage"]
    return compact.drop(columns=["Snapshot_Date_Parsed", "Snapshot_Stage_Order"], errors="ignore")


def player_history(history, query, team=None):
    if history.empty:
        return history
    target = normalize_text(query)
    candidates = history.copy()
    if team:
        candidates = candidates[candidates["Team Name"].astype(str).str.lower().eq(str(team).lower())]
    exact_names = candidates[candidates["Name"].apply(normalize_text).eq(target)]
    if exact_names.empty:
        exact_names = candidates[candidates["Name"].apply(normalize_text).str.contains(target, regex=False)]
    if exact_names.empty:
        return pd.DataFrame(columns=history.columns)
    if exact_names["PlayerIdKey"].nunique() > 1:
        first_id = exact_names.sort_values("Snapshot")["PlayerIdKey"].iloc[-1]
    else:
        first_id = exact_names["PlayerIdKey"].iloc[0]
    ph = history[history["PlayerIdKey"].eq(str(first_id))].sort_values("Snapshot")
    return compact_history_milestones(ph)


def latest_movement(history, min_latest_context=3.0):
    if history.empty:
        return pd.DataFrame()
    numeric_cols = ["Current_Value", "Context_Value", "Skill_Score", "YTD_Value", "Trend_Trade_Adjustment"]
    rows = []
    for _, group in history.groupby("PlayerIdKey"):
        ordered = compact_history_milestones(group).sort_values("Snapshot")
        if len(ordered) < 2:
            continue
        prev = ordered.iloc[-2]
        latest = ordered.iloc[-1]
        row = latest.to_dict()
        for col in numeric_cols:
            row[f"{col}_Delta"] = float(latest[col]) - float(prev[col]) if pd.notna(latest[col]) and pd.notna(prev[col]) else pd.NA
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    movement = pd.DataFrame(rows)
    if "Context_Value" in movement.columns and min_latest_context is not None:
        movement = movement[pd.to_numeric(movement["Context_Value"], errors="coerce") >= min_latest_context]
    return movement.sort_values("Context_Value_Delta", ascending=False).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Build player trend history from cached snapshots.")
    parser.add_argument("command", nargs="?", default="movers", choices=["history", "movers"])
    parser.add_argument("--player", default=None)
    parser.add_argument("--team", default=None)
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_DIR))
    args = parser.parse_args()

    history = build_trend_history(args.cache_root)
    if args.command == "history":
        if not args.player:
            print(history.to_string(index=False))
            return
        print(player_history(history, args.player, team=args.team).to_string(index=False))
        return

    movers = latest_movement(history)
    cols = [
        "Snapshot", "Team Name", "Name", "Positions", "Current_Value", "Context_Value",
        "Context_Value_Delta", "Skill_Score_Delta", "Trend_Label", "Trend_Notes",
    ]
    cols = [col for col in cols if col in movers.columns]
    print(movers.head(25)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
