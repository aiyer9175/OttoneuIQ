import os
import sys

import numpy as np


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
sys.path.insert(0, os.path.abspath(DATA_DIR))

from auction_lab import run_bandit_training
from bandit_policy import LinUCBBidPolicy


def test_linucb_policy_updates_action_parameters():
    policy = LinUCBBidPolicy(alpha=0.1, feature_dim=3, seed=1)
    context = np.array([1.0, 0.5, 0.0])
    decision = policy.choose(context)

    policy.observe(decision, reward=2.0, metadata={"Player": "Test Player"})

    summary = policy.action_summary()
    assert len(policy.observations) == 1
    assert summary["Decisions"].sum() == 1
    assert summary.iloc[0]["Avg_Reward"] == 2.0


def test_bandit_training_records_decisions():
    team_runs, strategy_summary, action_summary, decisions = run_bandit_training(
        sim_count=1,
        batters=os.path.join(DATA_DIR, "batters_auctioncalc.csv"),
        pitchers=os.path.join(DATA_DIR, "pitchers_auctioncalc.csv"),
        prospects=os.path.join(DATA_DIR, "Baseball Composite Prospect List 2026 - List.csv"),
        seed=11,
    )

    assert len(team_runs) == 12
    assert strategy_summary["Team_Sims"].sum() == 12
    assert not action_summary.empty
    assert not decisions.empty
    assert {"Action", "Reward", "Player", "Price", "Value"}.issubset(decisions.columns)
