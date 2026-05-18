import os
import sys

import pandas as pd


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
sys.path.insert(0, os.path.abspath(DATA_DIR))

from post_auction import assign_active_slots, load_simulated_rosters, merge_player_context, money_to_float, resolve_team_name


def test_money_to_float_handles_salary_strings():
    assert money_to_float("$12.50") == 12.5
    assert money_to_float("$1,234") == 1234.0
    assert money_to_float("") == 0.0


def test_merge_uses_player_id_before_name_accents():
    roster = pd.DataFrame([{
        "Name": "Jeremy Pena",
        "NameKey": "jeremy pena",
        "PlayerIdKey": "21636",
        "Salary": 17.0,
        "Roster_Positions": ["SS"],
    }])
    ros = pd.DataFrame([{
        "NameKey": "jeremy peña",
        "NameASCIIKey": "jeremy pena",
        "PlayerIdKey": "21636",
        "ROS_Dollars": 8.76,
        "ROS_Points": 500,
        "ROS_Positions": ["SS"],
    }])
    avg = pd.DataFrame([{
        "NameKey": "jeremy pena",
        "PlayerIdKey": "21636",
        "Avg_Salary": 10.39,
        "Median_Salary": 10.0,
        "Last10_Salary": 11.0,
        "Roster%": 100,
    }])
    prospects = pd.DataFrame(columns=["NameKey", "Prospect_Level", "Prospect_Rank"])

    merged = merge_player_context(roster, ros, avg, prospects)

    assert merged.loc[0, "ROS_Dollars"] == 8.76
    assert merged.loc[0, "Avg_Salary"] == 10.39
    assert merged.loc[0, "MLB_Level"]


def test_aaa_prospect_is_mlb_level_but_aa_prospect_is_not():
    roster = pd.DataFrame([
        {
            "Name": "AAA Prospect",
            "NameKey": "aaa prospect",
            "PlayerIdKey": None,
            "Salary": 1.0,
            "Roster_Positions": ["SS"],
        },
        {
            "Name": "AA Prospect",
            "NameKey": "aa prospect",
            "PlayerIdKey": None,
            "Salary": 1.0,
            "Roster_Positions": ["SS"],
        },
    ])
    ros = pd.DataFrame(columns=["NameKey", "NameASCIIKey", "PlayerIdKey", "ROS_Dollars", "ROS_Points", "ROS_Positions"])
    avg = pd.DataFrame(columns=["NameKey", "PlayerIdKey", "Avg_Salary", "Median_Salary", "Last10_Salary", "Roster%"])
    prospects = pd.DataFrame([
        {"NameKey": "aaa prospect", "Prospect_Level": "AAA", "Prospect_Rank": 1},
        {"NameKey": "aa prospect", "Prospect_Level": "AA", "Prospect_Rank": 2},
    ])

    merged = merge_player_context(roster, ros, avg, prospects)

    assert bool(merged.loc[0, "MLB_Level"])
    assert not bool(merged.loc[1, "MLB_Level"])


def test_resolve_team_name_accepts_case_insensitive_substring():
    report = pd.DataFrame({"Team Name": ["Contreras Bandits", "Brave Neu World"]})

    assert resolve_team_name(report, "contreras") == "Contreras Bandits"
    assert resolve_team_name(report, "BRAVE NEU WORLD") == "Brave Neu World"


def test_load_simulated_rosters_maps_live_auction_schema():
    simulated = pd.DataFrame([{
        "Team": "Team 1",
        "Player": "Auction Player",
        "Positions": "SS/2B",
        "Salary": 12,
    }])

    roster = load_simulated_rosters(simulated)

    assert roster.loc[0, "Team Name"] == "Team 1"
    assert roster.loc[0, "Name"] == "Auction Player"
    assert roster.loc[0, "Roster_Positions"] == ["SS", "2B"]
    assert roster.loc[0, "Salary"] == 12


def test_active_slot_assignment_prioritizes_rp_for_dual_eligible_pitchers():
    team = pd.DataFrame([
        {"Name": "Dual Arm", "MLB_Level": True, "Future_Value": 20.0, "Effective_Positions": ["SP", "RP"]},
        {"Name": "Starter Only", "MLB_Level": True, "Future_Value": 10.0, "Effective_Positions": ["SP"]},
        {"Name": "Bad Reliever", "MLB_Level": True, "Future_Value": -50.0, "Effective_Positions": ["RP"]},
    ])

    assigned = assign_active_slots(team)
    slots = dict(zip(assigned["Name"], assigned["Active_Slot"]))

    assert slots["Dual Arm"] == "RP"
    assert slots["Starter Only"] == "SP"
