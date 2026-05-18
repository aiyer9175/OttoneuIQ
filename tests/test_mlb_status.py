import os
import sys


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
sys.path.insert(0, os.path.abspath(DATA_DIR))

from mlb_status import classify_transaction, latest_transactions_by_player, status_rows_for_players


def test_status_marks_active_roster_players_as_active():
    rows = status_rows_for_players(
        ["111"],
        active_index={
            "111": {
                "Player_Name": "Active Player",
                "Current_Team_ID": 147,
                "Current_Team": "New York Yankees",
                "Current_Roster_Type": "active",
            }
        },
        transactions_by_player={},
        fetched_at="2026-05-16T00:00:00+00:00",
    )

    row = rows.iloc[0]
    assert row["MLB_Status"] == "ACTIVE_MLB"
    assert row["Status_Flag"] == "ACTIVE"
    assert row["Current_Team"] == "New York Yankees"


def test_status_marks_optioned_players_as_sent_down():
    tx = {
        "date": "2026-05-15",
        "typeDesc": "Status Change",
        "description": "Test Player optioned to Scranton/Wilkes-Barre RailRiders.",
        "person": {"id": 222, "fullName": "Test Player"},
    }
    rows = status_rows_for_players(
        ["222"],
        active_index={},
        transactions_by_player={"222": tx},
        fetched_at="2026-05-16T00:00:00+00:00",
    )

    row = rows.iloc[0]
    assert row["MLB_Status"] == "MINORS"
    assert row["Status_Flag"] == "SENT_DOWN"
    assert "optioned" in row["Latest_Transaction_Description"]


def test_latest_transactions_by_player_keeps_most_recent_event():
    latest = latest_transactions_by_player([
        {"date": "2026-05-01", "person": {"id": 333}, "description": "recalled"},
        {"date": "2026-05-10", "person": {"id": 333}, "description": "optioned"},
    ])

    assert latest["333"]["date"] == "2026-05-10"
    assert classify_transaction(latest["333"]) == ("MINORS", "SENT_DOWN")
