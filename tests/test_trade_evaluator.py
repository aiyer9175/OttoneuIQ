import os
import sys

import pandas as pd


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
sys.path.insert(0, os.path.abspath(DATA_DIR))

from trade_evaluator import (
    build_waiver_wire_board,
    evaluate_trade,
    fetch_ottoneu_roster_export,
    parse_players,
    recommend_trade_packages,
)


def test_parse_players_accepts_comma_separated_input():
    assert parse_players("A, B, C") == ["A", "B", "C"]
    assert parse_players("") == []


def test_evaluate_trade_scores_incoming_minus_outgoing():
    values = pd.DataFrame([
        {
            "Team Name": "Team A",
            "Name": "Outgoing Player",
            "Positions": "SS",
            "Salary": 10.0,
            "Current_Value": 12.0,
            "Current_Surplus": 2.0,
            "Stock_Change": 0.0,
            "Value_Source": "Test",
        },
        {
            "Team Name": "Team B",
            "Name": "Incoming Player",
            "Positions": "OF",
            "Salary": 5.0,
            "Current_Value": 15.0,
            "Current_Surplus": 10.0,
            "Stock_Change": 3.0,
            "Value_Source": "Test",
        },
    ])

    result = evaluate_trade(
        values,
        team_a="Team A",
        sends=["Outgoing Player"],
        receives=["Incoming Player"],
        team_b="Team B",
    )

    assert result["value_delta"] == 3.0
    assert result["salary_delta"] == -5.0
    assert result["surplus_delta"] == 8.0
    assert result["verdict"] == "ACCEPT"


def test_recommend_trade_packages_finds_single_and_multi_player_options():
    values = pd.DataFrame([
        {"Team Name": "Mine", "Name": "Mid Bat", "Positions": "OF", "Salary": 5.0, "Current_Value": 9.0, "Current_Surplus": 4.0, "Stock_Change": 1.0},
        {"Team Name": "Mine", "Name": "Depth Arm", "Positions": "SP", "Salary": 2.0, "Current_Value": 4.0, "Current_Surplus": 2.0, "Stock_Change": 0.0},
        {"Team Name": "Mine", "Name": "Bench Bat", "Positions": "1B", "Salary": 1.0, "Current_Value": 3.0, "Current_Surplus": 2.0, "Stock_Change": 0.0},
        {"Team Name": "Other", "Name": "Target Player", "Positions": "SS", "Salary": 7.0, "Current_Value": 12.0, "Current_Surplus": 5.0, "Stock_Change": 2.0},
    ])

    packages = recommend_trade_packages(values, "Mine", "Target Player", target_team="Other", max_package_size=2)

    assert not packages.empty
    assert packages.iloc[0]["Target"] == "Target Player (SS)"
    assert packages["Package"].str.contains("Mid Bat").any()
    assert packages["Package_Size"].max() <= 2


def test_recommend_trade_packages_accepts_incoming_package():
    values = pd.DataFrame([
        {"Team Name": "Mine", "Name": "Good Bat", "Positions": "OF", "Salary": 8.0, "Current_Value": 14.0, "Current_Surplus": 6.0, "Stock_Change": 1.0},
        {"Team Name": "Mine", "Name": "Useful Arm", "Positions": "SP", "Salary": 4.0, "Current_Value": 8.0, "Current_Surplus": 4.0, "Stock_Change": 0.0},
        {"Team Name": "Other", "Name": "Target One", "Positions": "SS", "Salary": 7.0, "Current_Value": 12.0, "Current_Surplus": 5.0, "Stock_Change": 2.0},
        {"Team Name": "Other", "Name": "Target Two", "Positions": "RP", "Salary": 2.0, "Current_Value": 6.0, "Current_Surplus": 4.0, "Stock_Change": 1.0},
    ])

    packages = recommend_trade_packages(
        values,
        "Mine",
        "Target One, Target Two",
        target_team="Other",
        max_package_size=2,
    )

    assert not packages.empty
    assert packages.iloc[0]["Target"] == "Target One (SS) + Target Two (RP)"
    assert packages.iloc[0]["Receive_Value"] == 18.0


def test_recommend_trade_packages_reports_opponent_context():
    values = pd.DataFrame([
        {"Team Name": "Mine", "Name": "Helpful Catcher", "Positions": "C", "Active_Slot": "Reserve", "Salary": 2.0, "Current_Value": 8.0, "Current_Surplus": 6.0, "Stock_Change": 1.0},
        {"Team Name": "Mine", "Name": "Useful Arm", "Positions": "SP", "Active_Slot": "SP", "Salary": 4.0, "Current_Value": 8.0, "Current_Surplus": 4.0, "Stock_Change": 0.0},
        {"Team Name": "Other", "Name": "Weak Catcher", "Positions": "C", "Active_Slot": "C", "Salary": 1.0, "Current_Value": 1.0, "Current_Surplus": 0.0, "Stock_Change": 0.0},
        {"Team Name": "Other", "Name": "Target Arm", "Positions": "SP", "Active_Slot": "SP", "Salary": 10.0, "Current_Value": 9.0, "Current_Surplus": -1.0, "Stock_Change": 0.0},
    ])

    packages = recommend_trade_packages(values, "Mine", "Target Arm", target_team="Other", max_package_size=1)

    catcher_row = packages[packages["Package"].str.contains("Helpful Catcher")].iloc[0]
    assert "fills opponent need" in catcher_row["Opponent_Context"]
    assert catcher_row["Opponent_Salary_Delta"] == -8.0


def test_waiver_wire_board_excludes_rostered_players_and_scores_available(tmp_path):
    rosters = tmp_path / "rosters.csv"
    hitters = tmp_path / "hitters.csv"
    pitchers = tmp_path / "pitchers.csv"
    avg = tmp_path / "avg.csv"
    stock = tmp_path / "stock.csv"
    status = tmp_path / "status.csv"

    pd.DataFrame([{
        "TeamID": 1, "Team Name": "Mine", "ottoneu ID": 1, "FG MajorLeagueID": 100,
        "FG MinorLeagueID": "", "Name": "Rostered Guy", "MLB Team": "AAA",
        "Position(s)": "OF", "Salary": "$1",
    }]).to_csv(rosters, index=False)
    base_cols = ["Name", "Team", "POS", "ADP", "PA", "rPTS", "PTS", "aPOS", "Dollars", "NameASCII", "PlayerId", "MLBAMID"]
    pd.DataFrame([
        ["Rostered Guy", "AAA", "OF", 1, 100, 100, 100, 1, 8, "Rostered Guy", 100, 1000],
        ["Available Bat", "BBB", "SS/OF", 1, 100, 120, 120, 1, 10, "Available Bat", 200, 2000],
    ], columns=base_cols).to_csv(hitters, index=False)
    p_cols = ["Name", "Team", "POS", "ADP", "IP", "rPTS", "PTS", "aPOS", "Dollars", "NameASCII", "PlayerId", "MLBAMID"]
    pd.DataFrame([
        ["Available Arm", "CCC", "SP", 1, 50, 90, 90, 1, 6, "Available Arm", 300, 3000],
    ], columns=p_cols).to_csv(pitchers, index=False)
    pd.DataFrame([
        {"Name": "Available Bat", "FG MajorLeagueID": 200, "Avg Salary": "$2", "Median Salary": "$1", "Last 10": "$3", "Roster%": 10},
    ]).to_csv(avg, index=False)
    pd.DataFrame([
        {"PlayerIdKey": "200", "MLB_Stock_Change": 3, "YTD_Value": 8, "YTD_ROS_Gap": 2, "Skill_Score": .7, "Role_Change": "STABLE", "Stock_Label": "Riser", "Confidence_Label": "High"},
    ]).to_csv(stock, index=False)
    pd.DataFrame(columns=["MLBAMIDKey", "MLB_Status", "Status_Flag"]).to_csv(status, index=False)

    board = build_waiver_wire_board(rosters, hitters, pitchers, avg, stock, status)

    assert "Rostered Guy" not in set(board["Name"])
    assert "Available Bat" in set(board["Name"])
    assert board.iloc[0]["Composite_Score"] >= board.iloc[-1]["Composite_Score"]


def test_fetch_ottoneu_roster_export_writes_league_csv(tmp_path):
    calls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b"TeamID,Team Name,Name\n1,Test,Player\n"

    def opener(request, timeout):
        calls.append((request.full_url, timeout))
        return Response()

    output = tmp_path / "rosters.csv"
    fetch_ottoneu_roster_export("1919", output, opener=opener)

    assert output.read_text().startswith("TeamID,Team Name")
    assert calls[0][0] == "https://ottoneu.fangraphs.com/1919/rosterexport?csv=1"
