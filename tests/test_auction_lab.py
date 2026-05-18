import os
import sys


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
sys.path.insert(0, os.path.abspath(DATA_DIR))

from auction_lab import classify_strategy, run_auction_sims


def test_classify_strategy_identifies_common_shapes():
    assert classify_strategy({
        "Prospects": 2,
        "Prospect_Salary": 4,
        "Stars_40": 4,
        "Cheap_5": 25,
        "Mid_6_25": 8,
    }) == "extreme studs/duds"

    assert classify_strategy({
        "Prospects": 11,
        "Prospect_Salary": 11,
        "Stars_40": 1,
        "Cheap_5": 20,
        "Mid_6_25": 10,
    }) == "prospect-heavy"

    assert classify_strategy({
        "Prospects": 2,
        "Prospect_Salary": 4,
        "Stars_40": 1,
        "Cheap_5": 8,
        "Mid_6_25": 20,
    }) == "balanced depth"


def test_run_auction_sims_returns_team_strategy_rows():
    team_runs, strategy_summary, metric_summary = run_auction_sims(
        sim_count=1,
        batters=os.path.join(DATA_DIR, "batters_auctioncalc.csv"),
        pitchers=os.path.join(DATA_DIR, "pitchers_auctioncalc.csv"),
        prospects=os.path.join(DATA_DIR, "Baseball Composite Prospect List 2026 - List.csv"),
        seed=7,
    )

    assert len(team_runs) == 12
    assert strategy_summary["Team_Sims"].sum() == 12
    assert set(["Strategy", "Total_Surplus"]).issubset(metric_summary.columns)
    assert team_runs["Open_Active"].eq(0).all()
