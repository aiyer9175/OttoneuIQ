import os
import sys

import pandas as pd


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
sys.path.insert(0, os.path.abspath(DATA_DIR))

from mlb_stock import build_mlb_stock


def write_auction(path, rows, playing_time_col):
    base_cols = ["Name", "Team", "POS", "ADP", playing_time_col, "rPTS", "PTS", "aPOS", "Dollars", "NameASCII", "PlayerId", "MLBAMID"]
    pd.DataFrame(rows, columns=base_cols)[base_cols].to_csv(path, index=False)


def test_mlb_stock_detects_rp_to_sp_role_change(tmp_path):
    h_pre = tmp_path / "h_pre.csv"
    h_ros = tmp_path / "h_ros.csv"
    p_pre = tmp_path / "p_pre.csv"
    p_ros = tmp_path / "p_ros.csv"
    empty_h_ytd = tmp_path / "h_ytd.csv"
    empty_h_sc = tmp_path / "h_sc.csv"
    p_ytd = tmp_path / "p_ytd.csv"
    empty_p_sc = tmp_path / "p_sc.csv"
    empty_rp = tmp_path / "rp.csv"

    write_auction(h_pre, [], "PA")
    write_auction(h_ros, [], "PA")
    write_auction(p_pre, [{
        "Name": "Role Guy", "Team": "AAA", "POS": "RP", "ADP": 1, "IP": 60,
        "rPTS": 300, "PTS": 1, "aPOS": 1, "Dollars": 6, "NameASCII": "Role Guy",
        "PlayerId": 1, "MLBAMID": 100,
    }], "IP")
    write_auction(p_ros, [{
        "Name": "Role Guy", "Team": "AAA", "POS": "SP", "ADP": 1, "IP": 120,
        "rPTS": 700, "PTS": 1, "aPOS": 1, "Dollars": 18, "NameASCII": "Role Guy",
        "PlayerId": 1, "MLBAMID": 100,
    }], "IP")
    pd.DataFrame(columns=["PlayerId", "MLBAMID"]).to_csv(empty_h_ytd, index=False)
    pd.DataFrame(columns=["player_id"]).to_csv(empty_h_sc, index=False)
    pd.DataFrame([{
        "Name": "Role Guy", "Team": "AAA", "W": 1, "L": 0, "SV": 0, "G": 7, "GS": 6,
        "IP": 35, "K/9": 10, "BB/9": 3, "HR/9": 1, "BABIP": .300, "LOB%": .75,
        "GB%": .45, "HR/FB": .10, "vFA (pi)": 95, "ERA": 3, "xERA": 3, "FIP": 3,
        "xFIP": 3, "WAR": 1, "NameASCII": "Role Guy", "PlayerId": 1, "MLBAMID": 100,
    }]).to_csv(p_ytd, index=False)
    pd.DataFrame(columns=["player_id"]).to_csv(empty_p_sc, index=False)
    pd.DataFrame(columns=["PlayerId", "MLBAMID"]).to_csv(empty_rp, index=False)

    stock = build_mlb_stock(
        str(h_pre), str(h_ros), str(p_pre), str(p_ros),
        str(empty_h_ytd), str(empty_h_sc), str(p_ytd), str(empty_p_sc), str(empty_rp),
    )

    row = stock.iloc[0]
    assert row["Role_Change"] == "RP_TO_SP"
    assert row["Stock_Label"] == "Role-Driven Riser"
    assert row["MLB_Stock_Change"] > 10


def test_negative_backend_move_is_capped_without_role_change(tmp_path):
    h_pre = tmp_path / "h_pre.csv"
    h_ros = tmp_path / "h_ros.csv"
    p_pre = tmp_path / "p_pre.csv"
    p_ros = tmp_path / "p_ros.csv"
    empty = tmp_path / "empty.csv"
    empty_sc = tmp_path / "empty_sc.csv"

    write_auction(h_pre, [], "PA")
    write_auction(h_ros, [], "PA")
    write_auction(p_pre, [{
        "Name": "Backend Arm", "Team": "AAA", "POS": "SP/RP", "ADP": 1, "IP": 20,
        "rPTS": 100, "PTS": 1, "aPOS": 1, "Dollars": -80, "NameASCII": "Backend Arm",
        "PlayerId": 2, "MLBAMID": 200,
    }], "IP")
    write_auction(p_ros, [{
        "Name": "Backend Arm", "Team": "AAA", "POS": "SP/RP", "ADP": 1, "IP": 70,
        "rPTS": 400, "PTS": 1, "aPOS": 1, "Dollars": 2, "NameASCII": "Backend Arm",
        "PlayerId": 2, "MLBAMID": 200,
    }], "IP")
    pd.DataFrame(columns=["PlayerId", "MLBAMID"]).to_csv(empty, index=False)
    pd.DataFrame(columns=["player_id"]).to_csv(empty_sc, index=False)

    stock = build_mlb_stock(
        str(h_pre), str(h_ros), str(p_pre), str(p_ros),
        str(empty), str(empty_sc), str(empty), str(empty_sc), str(empty),
    )

    row = stock.iloc[0]
    assert row["Projection_Change"] == 82
    assert row["MLB_Stock_Change"] <= 12
