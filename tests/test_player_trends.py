import os
import sys

import pandas as pd


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
sys.path.insert(0, os.path.abspath(DATA_DIR))

from player_trends import (
    build_player_trend_table,
    load_sp_rank_overrides,
    player_role,
    primary_position,
    trend_trade_adjustment,
    trend_waterfall_rows,
)


def make_values():
    return pd.DataFrame([
        {
            "Team Name": "Contender",
            "Name": "Elite Second Baseman",
            "Positions": "2B",
            "Salary": 42,
            "Current_Value": 34.0,
            "Current_Surplus": -8.0,
            "Stock_Change": -4.0,
            "PlayerIdKey": "1",
        },
        {
            "Team Name": "Rebuilder",
            "Name": "Skills Breakout",
            "Positions": "2B/3B",
            "Salary": 5,
            "Current_Value": 13.0,
            "Current_Surplus": 8.0,
            "Stock_Change": 9.0,
            "PlayerIdKey": "2",
        },
        {
            "Team Name": "Team C",
            "Name": "Third Base Only",
            "Positions": "3B",
            "Salary": 4,
            "Current_Value": 9.0,
            "Current_Surplus": 5.0,
            "Stock_Change": 1.0,
            "PlayerIdKey": "3",
        },
        {
            "Team Name": "Team D",
            "Name": "Ground Truth Starter",
            "Positions": "SP",
            "Salary": 20,
            "Current_Value": 30.0,
            "Current_Surplus": 10.0,
            "Stock_Change": 3.0,
            "PlayerIdKey": "4",
            "MLBAMIDKey": "111",
        },
        {
            "Team Name": "Team E",
            "Name": "Fallback Starter",
            "Positions": "SP",
            "Salary": 10,
            "Current_Value": 20.0,
            "Current_Surplus": 10.0,
            "Stock_Change": 2.0,
            "PlayerIdKey": "5",
            "MLBAMIDKey": "222",
        },
    ])


def make_stock():
    return pd.DataFrame([
        {
            "PlayerIdKey": "1",
            "Player_Type": "hitter",
            "Preseason_Value": 38.0,
            "ROS_Value": 34.0,
            "Projection_Change": -4.0,
            "YTD_Value": 10.0,
            "YTD_ROS_Gap": -24.0,
            "Skill_Score": 0.32,
            "Sample_Confidence": 0.90,
            "Role_Change": "STABLE",
            "Stock_Label": "Skills-Backed Faller",
            "MLB_Stock_Change": -5.5,
            "YTD_PA": 180,
            "YTD_wRC+": 70,
        },
        {
            "PlayerIdKey": "2",
            "Player_Type": "hitter",
            "Preseason_Value": 2.0,
            "ROS_Value": 13.0,
            "Projection_Change": 11.0,
            "YTD_Value": 30.0,
            "YTD_ROS_Gap": 17.0,
            "Skill_Score": 0.72,
            "Sample_Confidence": 0.85,
            "Role_Change": "STABLE",
            "Stock_Label": "Skills-Backed Riser",
            "MLB_Stock_Change": 13.0,
            "YTD_PA": 170,
            "YTD_wRC+": 145,
        },
        {
            "PlayerIdKey": "3",
            "Player_Type": "hitter",
            "Preseason_Value": 8.0,
            "ROS_Value": 9.0,
            "Projection_Change": 1.0,
            "YTD_Value": 9.0,
            "YTD_ROS_Gap": 0.0,
            "Skill_Score": 0.50,
            "Sample_Confidence": 0.80,
            "Role_Change": "STABLE",
            "Stock_Label": "Stable",
            "MLB_Stock_Change": 1.0,
            "YTD_PA": 160,
            "YTD_wRC+": 100,
        },
        {
            "PlayerIdKey": "4",
            "MLBAMIDKey": "111",
            "Player_Type": "pitcher",
            "Preseason_Value": 27.0,
            "ROS_Value": 30.0,
            "Projection_Change": 3.0,
            "YTD_Value": 18.0,
            "YTD_ROS_Gap": -12.0,
            "Skill_Score": 0.70,
            "Sample_Confidence": 0.80,
            "Role_Change": "STABLE_SP",
            "Stock_Label": "Stable",
            "MLB_Stock_Change": 3.0,
            "YTD_G": 8,
            "YTD_IP": 48,
            "YTD_GS": 8,
        },
        {
            "PlayerIdKey": "5",
            "MLBAMIDKey": "222",
            "Player_Type": "pitcher",
            "Preseason_Value": 18.0,
            "ROS_Value": 20.0,
            "Projection_Change": 2.0,
            "YTD_Value": 16.0,
            "YTD_ROS_Gap": -4.0,
            "Skill_Score": 0.60,
            "Sample_Confidence": 0.80,
            "Role_Change": "STABLE_SP",
            "Stock_Label": "Stable",
            "MLB_Stock_Change": 2.0,
            "YTD_G": 7,
            "YTD_IP": 42,
            "YTD_GS": 7,
        },
    ])


def test_elite_bad_start_is_dampened_not_repriced():
    trends = build_player_trend_table(make_values(), make_stock())
    row = trends[trends["Name"].eq("Elite Second Baseman")].iloc[0]

    assert row["Trend_Trade_Adjustment"] < 0
    assert row["Trend_Trade_Adjustment"] > -4
    assert row["Context_Value"] > 30
    assert row["Trend_Label"] == "Track Record Hold"
    assert "strong ROS anchor" in row["Trend_Notes"]


def test_skills_backed_breakout_gets_context_bump():
    trends = build_player_trend_table(make_values(), make_stock())
    row = trends[trends["Name"].eq("Skills Breakout")].iloc[0]

    assert row["Trend_Trade_Adjustment"] > 2
    assert row["Context_Value"] > row["Current_Value"]
    assert row["Trend_Label"] == "Skills-Backed Riser"
    assert "skills support production" in row["Trend_Notes"]


def test_projection_skeptical_skills_breakout_can_clear_opening_day_style_value():
    row = {
        "Current_Value": 6.4,
        "Skill_Score": 0.746,
        "Sample_Confidence": 1.0,
        "YTD_ROS_Gap": 25.2,
    }

    assert trend_trade_adjustment(row) >= 6


def test_results_without_skill_support_stays_conservative():
    row = {
        "Current_Value": 1.7,
        "Skill_Score": 0.236,
        "Sample_Confidence": 0.99,
        "YTD_ROS_Gap": 8.3,
    }

    assert trend_trade_adjustment(row) < 0


def test_waterfall_includes_context_value_row():
    trends = build_player_trend_table(make_values(), make_stock())
    row = trends[trends["Name"].eq("Skills Breakout")].iloc[0]

    waterfall = trend_waterfall_rows(row)

    assert list(waterfall["Component"])[0] == "ROS Value"
    assert list(waterfall["Component"])[-1] == "Context Value"


def test_primary_position_and_role_are_derived():
    trends = build_player_trend_table(make_values(), make_stock())
    row = trends[trends["Name"].eq("Skills Breakout")].iloc[0]

    assert primary_position("2B/3B") == "2B"
    assert row["Primary_Position"] == "2B"
    assert row["Position_Rank"] >= 1
    assert row["Player_Role"] in {"Everyday Bat", "Regular Role", "Projected Bat", "Core Bat"}


def test_eligible_position_ranks_include_multi_position_eligibility():
    trends = build_player_trend_table(make_values(), make_stock())
    row = trends[trends["Name"].eq("Skills Breakout")].iloc[0]

    assert "2B #" in row["Eligible_Position_Ranks"]
    assert "3B #1" in row["Eligible_Position_Ranks"]
    assert "DH #" not in row["Eligible_Position_Ranks"]


def test_pitcher_role_detects_closer():
    row = {
        "Positions": "RP",
        "Player_Type": "pitcher",
        "Current_Value": 12,
        "YTD_IP": 18,
        "YTD_GS": 0,
        "YTD_SV": 8,
    }

    assert player_role(row) == "Closer"


def test_sp_rank_override_uses_pitch_report_rank(tmp_path):
    ranks = tmp_path / "sp_ranks.csv"
    ranks.write_text("Eno,Name,MLBAM id\n7,Ground Truth Starter,111\n")

    overrides = load_sp_rank_overrides(ranks)
    trends = build_player_trend_table(make_values(), make_stock(), sp_ranks=ranks)
    ground_truth = trends[trends["Name"].eq("Ground Truth Starter")].iloc[0]
    fallback = trends[trends["Name"].eq("Fallback Starter")].iloc[0]

    assert overrides == {"111": 7}
    assert ground_truth["Position_Rank"] == 7
    assert "SP #7" in ground_truth["Eligible_Position_Ranks"]
    assert "SP #" in fallback["Eligible_Position_Ranks"]
    assert "SP #7" not in fallback["Eligible_Position_Ranks"]


def test_sp_rp_zero_starts_is_not_rotation_starter():
    row = {
        "Positions": "SP/RP",
        "Player_Type": "pitcher",
        "Current_Value": 23,
        "YTD_G": 12,
        "YTD_IP": 16.1,
        "YTD_GS": 0,
        "YTD_SV": 0,
        "Skill_Score": 0.93,
    }

    assert player_role(row) == "High-Leverage RP"


def test_rotation_starter_tiers_use_starts_and_skill():
    ace = {
        "Positions": "SP",
        "Player_Type": "pitcher",
        "Current_Value": 32,
        "YTD_G": 9,
        "YTD_IP": 54,
        "YTD_GS": 9,
        "YTD_SV": 0,
        "Skill_Score": 0.70,
    }
    back_end = {
        "Positions": "SP",
        "Player_Type": "pitcher",
        "Current_Value": 4,
        "YTD_G": 7,
        "YTD_IP": 35,
        "YTD_GS": 7,
        "YTD_SV": 0,
        "Skill_Score": 0.46,
    }

    assert player_role(ace) == "Ace / SP1"
    assert player_role(back_end) == "Back-End Starter"
