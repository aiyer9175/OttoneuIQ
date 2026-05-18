import os
import sys

import pandas as pd


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
sys.path.insert(0, os.path.abspath(DATA_DIR))

from roster_audit import build_roster_audit, position_matches


def row(team, name, positions, slot, salary, value, prospect=False):
    return {
        "Team Name": team,
        "Name": name,
        "Positions": positions,
        "Active_Slot": slot,
        "Salary": salary,
        "Current_Value": value,
        "Current_Surplus": value - salary,
        "Is_Prospect": prospect,
        "Stock_Label": "Standard",
    }


def test_position_matching_handles_flexible_slots():
    assert position_matches({"Positions": "SS"}, "MI")
    assert position_matches({"Positions": "3B"}, "CI")
    assert position_matches({"Positions": "OF"}, "UTIL")
    assert not position_matches({"Positions": "RP"}, "SP")


def test_roster_audit_surfaces_needs_and_trade_ideas():
    values = pd.DataFrame([
        row("Mine", "Weak Catcher", "C", "C", 2, 1),
        row("Mine", "Reserve Outfielder", "OF", "Reserve", 4, 12),
        row("Mine", "Prospect Chip", "SS", "Reserve", 3, 9, prospect=True),
        row("Other", "Strong Catcher", "C", "C", 10, 18),
        row("Other", "Average OF", "OF", "OF", 8, 8),
    ])

    audit = build_roster_audit(values, "Mine")

    needs = audit["positions"][audit["positions"]["Need_Level"].isin(["Major need", "Need"])]
    assert "C" in set(needs["Position"])
    assert "Reserve Outfielder" in set(audit["trade_chips"]["Name"])
    assert not audit["trade_ideas"].empty
    assert "Strong Catcher" in set(audit["trade_ideas"]["Target"])


def test_position_audit_floors_replacement_value_for_extreme_negative_projection():
    values = pd.DataFrame([
        row("Mine", "Useful Reliever", "RP", "RP", 1, 6),
        row("Mine", "Bad Projection", "RP", "RP", 1, -100),
        row("Other", "Average Reliever", "RP", "RP", 1, 5),
    ])

    audit = build_roster_audit(values, "Mine")
    rp = audit["positions"][audit["positions"]["Position"].eq("RP")].iloc[0]

    assert rp["Raw_Team_Value"] == -94
    assert rp["Team_Value"] == 1


def test_trade_ideas_can_build_two_for_one_packages_and_avoid_unreachable_superstars():
    values = pd.DataFrame([
        row("Mine", "Weak Third Baseman", "3B", "3B", 2, 1),
        row("Mine", "Good Outfielder", "OF", "Reserve", 4, 12),
        row("Mine", "Good Prospect", "SS", "Reserve", 3, 10, prospect=True),
        row("Other", "Useful Third Baseman", "3B", "3B", 8, 20),
        row("Other", "Superstar Third Baseman", "3B", "3B", 30, 55),
    ])

    audit = build_roster_audit(values, "Mine")
    ideas = audit["trade_ideas"]

    assert "Good Outfielder + Good Prospect" in set(ideas["Trade_Chip"])
    assert "Useful Third Baseman" in set(ideas["Target"])
    assert "Superstar Third Baseman" not in set(ideas["Target"])
