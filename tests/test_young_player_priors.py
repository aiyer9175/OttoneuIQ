import os
import sys

import pandas as pd


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
sys.path.insert(0, os.path.abspath(DATA_DIR))

from young_player_priors import add_ps_priors, load_ps_priors


def test_load_ps_priors_takes_best_player_year(tmp_path):
    path_2024 = tmp_path / "2024_ps_hitters.csv"
    path_2025 = tmp_path / "2025_ps_hitters.csv"
    path_2024.write_text("Name,Org,Age,PS Score,PA\nYoung Bat,MIL,21,0.72,200\n")
    path_2025.write_text("Name,Org,Age,PS Score,PA\nYoung Bat,MIL,22,0.96,160\n")

    priors = load_ps_priors([
        (path_2024, 2024, "hitter"),
        (path_2025, 2025, "hitter"),
    ])

    row = priors.iloc[0]
    assert row["PS_Best_Score"] == 0.96
    assert row["PS_Best_Year"] == 2025
    assert row["Prospect_Pedigree_Label"] == "Strong Prospect Track Record"


def test_add_ps_priors_matches_name_and_org(tmp_path):
    path = tmp_path / "2025_ps_hitters.csv"
    path.write_text("Name,Org,Age,PS Score,PA\nYoung Bat,MIL,22,0.96,160\n")
    priors = load_ps_priors([(path, 2025, "hitter")])
    trends = pd.DataFrame([{
        "Name": "Young Bat",
        "NameKey": "young bat",
        "MLB Team": "MIL",
    }])

    out = add_ps_priors(trends, priors)

    assert out.iloc[0]["Prospect_Pedigree_Label"] == "Strong Prospect Track Record"
