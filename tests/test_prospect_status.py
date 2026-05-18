import os
import sys

import pandas as pd


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
sys.path.insert(0, os.path.abspath(DATA_DIR))

from prospect_status import (
    add_recent_mlb_playing_time,
    apply_prospect_graduation,
    baseball_ip_to_outs,
    is_ungraduated_prospect,
)


def test_hitter_prospect_graduates_after_pa_threshold():
    assert is_ungraduated_prospect({"Is_Prospect": True, "Player_Type": "hitter", "YTD_PA": 130})
    assert not is_ungraduated_prospect({"Is_Prospect": True, "Player_Type": "hitter", "YTD_PA": 131})


def test_pitcher_prospect_graduates_after_ip_threshold():
    assert is_ungraduated_prospect({"Is_Prospect": True, "Player_Type": "pitcher", "YTD_IP": 50})
    assert not is_ungraduated_prospect({"Is_Prospect": True, "Player_Type": "pitcher", "YTD_IP": 50.1})


def test_pitcher_prospect_graduates_after_recent_career_ip_threshold():
    assert baseball_ip_to_outs(36.2) + baseball_ip_to_outs(21.1) == 174
    assert not is_ungraduated_prospect({
        "Is_Prospect": True,
        "Player_Type": "pitcher",
        "Career_MLB_Outs": baseball_ip_to_outs(36.2) + baseball_ip_to_outs(21.1),
    })


def test_missing_playing_time_keeps_prospect_label():
    assert is_ungraduated_prospect({"Is_Prospect": True, "Player_Type": "hitter", "YTD_PA": pd.NA})


def test_apply_prospect_graduation_preserves_listed_flag():
    df = pd.DataFrame([
        {"Name": "Graduated Hitter", "Is_Prospect": True, "Player_Type": "hitter", "YTD_PA": 150},
        {"Name": "Still Prospect", "Is_Prospect": True, "Player_Type": "pitcher", "YTD_IP": 12},
        {"Name": "Veteran", "Is_Prospect": False, "Player_Type": "hitter", "YTD_PA": 10},
    ])

    out = apply_prospect_graduation(df)

    assert out["Prospect_Listed"].tolist() == [True, True, False]
    assert out["Is_Prospect"].tolist() == [False, True, False]


def test_add_recent_mlb_playing_time_merges_career_totals():
    df = pd.DataFrame([{"Name": "Grant Taylor", "PlayerIdKey": "33927", "Is_Prospect": True}])
    playing_time = pd.DataFrame([{
        "PlayerIdKey": "33927",
        "Career_MLB_PA": 0,
        "Career_MLB_IP": 57.2,
        "Career_MLB_Outs": 173,
    }])

    out = add_recent_mlb_playing_time(df, playing_time=playing_time)

    assert out.loc[0, "Career_MLB_Outs"] == 173
