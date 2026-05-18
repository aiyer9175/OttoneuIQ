import os
import sys

import pandas as pd


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
sys.path.insert(0, os.path.abspath(DATA_DIR))

from prospect_status import apply_prospect_graduation, is_ungraduated_prospect


def test_hitter_prospect_graduates_after_pa_threshold():
    assert is_ungraduated_prospect({"Is_Prospect": True, "Player_Type": "hitter", "YTD_PA": 130})
    assert not is_ungraduated_prospect({"Is_Prospect": True, "Player_Type": "hitter", "YTD_PA": 131})


def test_pitcher_prospect_graduates_after_ip_threshold():
    assert is_ungraduated_prospect({"Is_Prospect": True, "Player_Type": "pitcher", "YTD_IP": 50})
    assert not is_ungraduated_prospect({"Is_Prospect": True, "Player_Type": "pitcher", "YTD_IP": 50.1})


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
