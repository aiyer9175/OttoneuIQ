import os
import sys

import pandas as pd


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
sys.path.insert(0, os.path.abspath(DATA_DIR))

from statcast_sources import normalize_expected_statcast, refresh_statcast_csvs


def test_normalize_hitter_expected_statcast_outputs_existing_schema():
    expected = pd.DataFrame([{
        "last_name, first_name": "Breakout, Test",
        "player_id": 123,
        "year": 2026,
        "pa": 100,
        "woba": .350,
        "est_woba": .410,
        "est_ba": .300,
        "est_slg": .550,
    }])
    contact = pd.DataFrame([{
        "player_id": 123,
        "brl_percent": 15.5,
        "ev95percent": 52.0,
    }])

    rows = normalize_expected_statcast(expected, contact, "hitter")
    row = rows.iloc[0]

    assert row["player_id"] == 123
    assert row["xwoba"] == .410
    assert row["xslg"] == .550
    assert row["barrel_batted_rate"] == 15.5
    assert row["hard_hit_percent"] == 52.0


def test_refresh_statcast_csvs_uses_injected_fetchers(tmp_path):
    expected = pd.DataFrame([{
        "last_name, first_name": "Pitcher, Test",
        "player_id": 456,
        "year": 2026,
        "pa": 120,
        "woba": .290,
        "est_woba": .270,
        "est_ba": .210,
        "est_slg": .320,
        "xera": 3.10,
    }])
    contact = pd.DataFrame([{
        "player_id": 456,
        "brl_percent": 5.0,
        "ev95percent": 34.0,
    }])
    fetchers = {
        "batter_expected": lambda year, min_pa: expected,
        "batter_contact": lambda year, min_bbe: contact,
        "pitcher_expected": lambda year, min_pa: expected,
        "pitcher_contact": lambda year, min_bbe: contact,
    }
    hitters_path = tmp_path / "hitters.csv"
    pitchers_path = tmp_path / "pitchers.csv"

    hitters, pitchers = refresh_statcast_csvs(
        hitters_path,
        pitchers_path,
        year=2026,
        fetchers=fetchers,
    )

    assert hitters_path.exists()
    assert pitchers_path.exists()
    assert len(hitters) == 1
    assert pitchers.iloc[0]["xera"] == 3.10
