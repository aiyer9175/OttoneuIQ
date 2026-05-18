import os
import sys


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
sys.path.insert(0, os.path.abspath(DATA_DIR))

from policy import player_policy_groups, policy_max_overpay, policy_multiplier


def test_player_policy_groups_can_stack_position_and_archetype():
    player = {"positions": ["SP"], "dollars": 45, "is_prospect": False}

    assert player_policy_groups(player) == ["SP", "STAR"]


def test_policy_multiplier_averages_group_parameters():
    player = {"positions": ["SP"], "dollars": 45, "is_prospect": False}
    policy = {
        "DEFAULT": {"bid_multiplier": 1.0, "nomination_multiplier": 1.0, "max_overpay": 5},
        "SP": {"bid_multiplier": 1.2, "nomination_multiplier": 1.1, "max_overpay": 10},
        "STAR": {"bid_multiplier": 1.0, "nomination_multiplier": 1.3, "max_overpay": 20},
    }

    assert round(policy_multiplier(player, policy, "bid_multiplier"), 2) == 1.1
    assert round(policy_multiplier(player, policy, "nomination_multiplier"), 2) == 1.2
    assert policy_max_overpay(player, policy) == 15
