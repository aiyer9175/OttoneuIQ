import os
import sys

import pandas as pd


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
sys.path.insert(0, os.path.abspath(DATA_DIR))

from prospect_updates import build_prospect_updates


def test_pipeline_prior_and_savant_evidence_create_posterior(tmp_path):
    pipeline = tmp_path / "pipeline.csv"
    composite = tmp_path / "composite.csv"
    avg = tmp_path / "avg.csv"
    savant = tmp_path / "AAA_hitters_YTD_test.csv"

    pd.DataFrame([{
        "Rank": 12,
        "Name": "Walker Jenkins",
        "MLBAMID": 805805,
        "Position": "OF",
        "Org": "Minnesota Twins",
        "Level": "AAA",
        "ETA": 2026,
        "Age": 21,
    }]).to_csv(pipeline, index=False)
    pd.DataFrame([{
        "Name": "Jenkins, Walker",
        "Team": "MIN",
        "Rank": 7,
        "Highest Level": "AAA",
        "mlbam": 805805,
    }]).to_csv(composite, index=False)
    pd.DataFrame([{
        "Name": "Walker Jenkins",
        "MLB Org": "MIN",
        "Avg Salary": "$5.00",
        "Last 10": "$6.00",
        "Roster%": 80,
    }]).to_csv(avg, index=False)
    pd.DataFrame([{
        "Name": "Walker Jenkins",
        "Org": "MIN",
        "Pos": "OF",
        "Age": 21,
        "PA": 150,
        "Pitches": 600,
        "PS Score": 1.0,
        "xwOBA": 0.8,
        "wOBA": 0.7,
    }]).to_csv(savant, index=False)

    updates = build_prospect_updates(
        pipeline_csv=str(pipeline),
        composite_csv=str(composite),
        avg_csv=str(avg),
        savant_files=[str(savant)],
    )

    row = updates.iloc[0]
    assert row["Pipeline_Rank"] == 12
    assert row["MLBAMID"] == 805805
    assert row["Prior_Value"] >= 15
    assert row["Updated_Prospect_Value"] > row["Prior_Value"]
    assert row["Confidence_Label"] == "High"


def test_aaa_starter_stuff_plus_strengthens_pitcher_update(tmp_path):
    pipeline = tmp_path / "pipeline.csv"
    composite = tmp_path / "composite.csv"
    avg = tmp_path / "avg.csv"
    savant = tmp_path / "AAA_pitchers_YTD_test.csv"
    stuff = tmp_path / "stuff_plus.csv"

    pd.DataFrame([{
        "Rank": 76,
        "Name": "Carlos Lagrange",
        "MLBAMID": 801739,
        "Position": "RHP",
        "Org": "New York Yankees",
        "Level": "AAA",
        "ETA": 2026,
        "Age": 22,
    }]).to_csv(pipeline, index=False)
    pd.DataFrame([{
        "Name": "Lagrange, Carlos",
        "Team": "NYY",
        "Rank": 79,
        "Highest Level": "AA",
        "mlbam": 801739,
    }]).to_csv(composite, index=False)
    pd.DataFrame([{
        "Name": "Carlos Lagrange",
        "MLB Org": "NYY",
        "Avg Salary": "$1.00",
        "Last 10": "$1.00",
        "Roster%": 20,
    }]).to_csv(avg, index=False)
    pd.DataFrame([{
        "Name": "Carlos Lagrange",
        "Org": "NYY",
        "Pos": "SP",
        "Age": 22,
        "PA": 145,
        "Pitches": 615,
        "PS Score": 0.94,
        "xwOBA": 0.338,
        "wOBA": 0.346,
        "K%.1": 30.3,
        "BB%.1": 13.1,
        "Whiff%.1": 34.2,
        "SwStr%.1": 14.5,
    }]).to_csv(savant, index=False)
    pd.DataFrame([{
        "season": 2026,
        "level": "AAA",
        "game_type": "R",
        "pitcher": 801739,
        "player_name": "Lagrange, Carlos",
        "n_pitches": 546,
        "P+": 98.08,
        "S+": 106.6,
        "L+": 86.76,
    }]).to_csv(stuff, index=False)

    without_stuff = build_prospect_updates(
        pipeline_csv=str(pipeline),
        composite_csv=str(composite),
        avg_csv=str(avg),
        savant_files=[str(savant)],
        stuff_plus_csv=None,
    ).iloc[0]
    with_stuff = build_prospect_updates(
        pipeline_csv=str(pipeline),
        composite_csv=str(composite),
        avg_csv=str(avg),
        savant_files=[str(savant)],
        stuff_plus_csv=str(stuff),
    ).iloc[0]

    assert with_stuff["Stuff_Plus"] == 106.6
    assert with_stuff["Stuff_Pitches"] == 546
    assert with_stuff["Prospect_Update"] > without_stuff["Prospect_Update"]
