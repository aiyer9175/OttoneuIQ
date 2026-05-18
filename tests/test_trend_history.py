import os
import sys

import pandas as pd


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
sys.path.insert(0, os.path.abspath(DATA_DIR))

from snapshots import PLAYER_TRENDS_FILE
from trend_history import build_trend_history, compact_history_milestones, latest_movement, player_history


def write_snapshot(cache_root, stamp, context_value, skill_score):
    path = cache_root / stamp
    path.mkdir(parents=True)
    pd.DataFrame([
        {
            "Team Name": "Team A",
            "Name": "Trend Player",
            "PlayerIdKey": "123",
            "Positions": "2B",
            "Salary": 5,
            "Current_Value": context_value - 1,
            "Context_Value": context_value,
            "Trend_Trade_Adjustment": 1,
            "Trend_Label": "Stable Context",
            "Trend_Sample": "stable sample",
            "YTD_Value": 10,
            "YTD_ROS_Gap": 2,
            "Projection_Change": 1,
            "Skill_Score": skill_score,
            "Sample_Confidence": 0.8,
            "Role_Change": "STABLE",
            "Stock_Label": "Stable",
            "Trend_Notes": "limited trend signal",
        }
    ]).to_csv(path / PLAYER_TRENDS_FILE, index=False)


def test_build_trend_history_reads_snapshot_series(tmp_path):
    write_snapshot(tmp_path, "20251231T235959Z", 10, 0.50)
    write_snapshot(tmp_path, "20260326T000000Z", 13, 0.62)

    history = build_trend_history(tmp_path)
    ph = player_history(history, "Trend Player")

    assert len(history) == 2
    assert len(ph) == 2
    assert list(ph["Context_Value"]) == [10, 13]


def test_latest_movement_compares_last_two_snapshots(tmp_path):
    write_snapshot(tmp_path, "20260326T000000Z", 10, 0.50)
    write_snapshot(tmp_path, "20260508T000000Z", 13, 0.62)

    movement = latest_movement(build_trend_history(tmp_path))

    assert len(movement) == 1
    assert movement.iloc[0]["Context_Value_Delta"] == 3
    assert round(movement.iloc[0]["Skill_Score_Delta"], 2) == 0.12


def test_player_history_compacts_to_milestone_snapshots(tmp_path):
    write_snapshot(tmp_path, "20251231T235959Z", 8, 0.45)
    write_snapshot(tmp_path, "20260326T000000Z", 10, 0.50)
    write_snapshot(tmp_path, "20260516T000000Z", 11, 0.55)
    write_snapshot(tmp_path, "20260516T163525Z", 12, 0.56)
    write_snapshot(tmp_path, "20260517T230801Z", 13, 0.60)
    write_snapshot(tmp_path, "20260517T232252Z", 14, 0.62)

    ph = player_history(build_trend_history(tmp_path), "Trend Player")

    assert list(ph["Snapshot_Label"]) == ["2025 end", "2026 start", "today"]
    assert list(ph["Context_Value"]) == [8, 10, 14]


def test_compact_history_milestones_keeps_empty_shape():
    empty = pd.DataFrame(columns=["Snapshot", "Snapshot_Date", "PlayerIdKey"])

    compact = compact_history_milestones(empty)

    assert compact.empty
