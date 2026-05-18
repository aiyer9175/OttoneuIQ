import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Pandas requires version")
import pandas as pd

from data_sources import DEFAULT_CACHE_DIR
from mlb_stock import (
    DEFAULT_HITTERS_PRESEASON,
    DEFAULT_PITCHERS_PRESEASON,
    load_auction_values,
)
from snapshots import PLAYER_TRENDS_FILE, PLAYER_VALUES_FILE
from value_engine import build_player_value_table


DEFAULT_BASELINE_TIMESTAMP = "20251231T235959Z"
DEFAULT_OPENING_2026_TIMESTAMP = "20260326T000000Z"


def load_preseason_value_pool(hitters_preseason=DEFAULT_HITTERS_PRESEASON, pitchers_preseason=DEFAULT_PITCHERS_PRESEASON):
    hitters = load_auction_values(hitters_preseason, "hitter", "Baseline")
    pitchers = load_auction_values(pitchers_preseason, "pitcher", "Baseline")
    pool = pd.concat([hitters, pitchers], ignore_index=True, sort=False)
    pool = pool.drop_duplicates("PlayerIdKey", keep="first")
    pool["Current_Value"] = pd.to_numeric(pool["Baseline_Value"], errors="coerce").fillna(0)
    pool["Context_Value"] = pool["Current_Value"]
    pool["Trend_Trade_Adjustment"] = 0.0
    pool["Trend_Label"] = "End-2025 Baseline"
    pool["Trend_Sample"] = "baseline"
    pool["Trend_Notes"] = "preseason auction value baseline"
    pool["Skill_Score"] = 0.5
    pool["Sample_Confidence"] = 0.0
    pool["Role_Change"] = "BASELINE"
    pool["Stock_Label"] = "Baseline"
    pool["YTD_Value"] = pd.NA
    pool["YTD_ROS_Gap"] = pd.NA
    pool["Projection_Change"] = 0.0
    return pool


def merge_current_roster_context(baseline, current_values=None):
    if current_values is None:
        current_values, _ = build_player_value_table()
    context_cols = ["PlayerIdKey", "Team Name", "Name", "Positions", "Salary", "Current_Surplus", "Is_Prospect"]
    context_cols = [col for col in context_cols if col in current_values.columns]
    current_context = current_values[context_cols].drop_duplicates("PlayerIdKey")
    merged = baseline.merge(current_context, on="PlayerIdKey", how="left", suffixes=("", "_Current"))
    merged["Name"] = merged["Name"].combine_first(merged["Baseline_Name"])
    merged["Positions"] = merged["Positions"].combine_first(merged["Baseline_POS"])
    merged["Team Name"] = merged["Team Name"].fillna("Free Agent / Unrostered")
    merged["Salary"] = pd.to_numeric(merged["Salary"], errors="coerce")
    merged["Current_Surplus"] = merged["Current_Surplus"].fillna(merged["Current_Value"] - merged["Salary"].fillna(0))
    return merged


def baseline_snapshot_rows(
    hitters_preseason=DEFAULT_HITTERS_PRESEASON,
    pitchers_preseason=DEFAULT_PITCHERS_PRESEASON,
    owned_only=True,
    label="End-2025 Baseline",
    notes="preseason auction value baseline",
    value_floor=None,
):
    baseline = load_preseason_value_pool(hitters_preseason, pitchers_preseason)
    rows = merge_current_roster_context(baseline)
    if owned_only:
        rows = rows[rows["Team Name"].ne("Free Agent / Unrostered")]
    if value_floor is not None:
        rows["Current_Value"] = pd.to_numeric(rows["Current_Value"], errors="coerce").clip(lower=value_floor)
        rows["Context_Value"] = pd.to_numeric(rows["Context_Value"], errors="coerce").clip(lower=value_floor)
        rows["Current_Surplus"] = rows["Current_Value"] - pd.to_numeric(rows["Salary"], errors="coerce").fillna(0)
    rows["Trend_Label"] = label
    rows["Trend_Notes"] = notes

    output_cols = [
        "Team Name", "Name", "PlayerIdKey", "Positions", "Salary",
        "Current_Value", "Context_Value", "Current_Surplus",
        "Trend_Trade_Adjustment", "Trend_Label", "Trend_Sample",
        "YTD_Value", "YTD_ROS_Gap", "Projection_Change", "Skill_Score",
        "Sample_Confidence", "Role_Change", "Stock_Label", "Trend_Notes",
        "Player_Type", "Baseline_Team", "Baseline_POS", "Baseline_PA", "Baseline_IP",
        "Baseline_rPTS", "Baseline_PTS",
    ]
    for col in output_cols:
        if col not in rows.columns:
            rows[col] = pd.NA
    return rows[output_cols].sort_values("Current_Value", ascending=False).reset_index(drop=True)


def write_historical_value_snapshot(
    cache_root=DEFAULT_CACHE_DIR,
    timestamp=DEFAULT_BASELINE_TIMESTAMP,
    hitters_preseason=DEFAULT_HITTERS_PRESEASON,
    pitchers_preseason=DEFAULT_PITCHERS_PRESEASON,
    owned_only=True,
    label="End-2025 Baseline",
    notes="preseason auction value baseline",
    source_label="preseason_fangraphs_auction_exports",
    value_floor=None,
):
    cache_dir = Path(cache_root) / timestamp
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = baseline_snapshot_rows(
        hitters_preseason,
        pitchers_preseason,
        owned_only=owned_only,
        label=label,
        notes=notes,
        value_floor=value_floor,
    )
    rows.to_csv(cache_dir / PLAYER_TRENDS_FILE, index=False, float_format="%.3f")
    rows.to_csv(cache_dir / PLAYER_VALUES_FILE, index=False, float_format="%.3f")
    manifest = {
        "created_at": timestamp,
        "source": source_label,
        "snapshot": {
            "created_at": timestamp,
            "source": source_label,
            "owned_only": bool(owned_only),
            "derived_files": {
                "player_values": {"file": PLAYER_VALUES_FILE, "rows": int(len(rows))},
                "player_trends": {"file": PLAYER_TRENDS_FILE, "rows": int(len(rows))},
            },
        },
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return cache_dir, manifest


def write_end_2025_mlb_snapshot(**kwargs):
    return write_historical_value_snapshot(**kwargs)


def main():
    parser = argparse.ArgumentParser(description="Build historical baseline snapshots.")
    parser.add_argument("command", choices=["end-2025-mlb", "end-2025-performance", "opening-2026"])
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--hitters-preseason", default=DEFAULT_HITTERS_PRESEASON)
    parser.add_argument("--pitchers-preseason", default=DEFAULT_PITCHERS_PRESEASON)
    parser.add_argument("--all-players", action="store_true", help="Include unrostered MLB value-pool players.")
    args = parser.parse_args()

    is_performance = args.command == "end-2025-performance"
    is_opening = args.command == "opening-2026"
    timestamp = args.timestamp
    if timestamp is None:
        timestamp = DEFAULT_OPENING_2026_TIMESTAMP if is_opening else DEFAULT_BASELINE_TIMESTAMP
    cache_dir, manifest = write_historical_value_snapshot(
        cache_root=args.cache_root,
        timestamp=timestamp,
        hitters_preseason=args.hitters_preseason,
        pitchers_preseason=args.pitchers_preseason,
        owned_only=not args.all_players,
        label=(
            "Opening Day 2026 Projection"
            if is_opening
            else "End-2025 Performance Baseline"
            if is_performance
            else "End-2025 Baseline"
        ),
        notes=(
            "2026 preseason FanGraphs projection dollar baseline"
            if is_opening
            else "2025 full-season FGPTS performance dollar baseline"
            if is_performance
            else "preseason auction value baseline"
        ),
        source_label=(
            "2026_opening_day_fangraphs_projection_exports"
            if is_opening
            else "2025_fangraphs_fgpts_performance_exports"
            if is_performance
            else "preseason_fangraphs_auction_exports"
        ),
        value_floor=0.0 if is_performance else None,
    )
    print(f"Built historical baseline snapshot: {cache_dir}")
    print(json.dumps(manifest["snapshot"], indent=2))


if __name__ == "__main__":
    main()
