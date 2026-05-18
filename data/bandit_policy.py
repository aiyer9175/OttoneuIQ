import random
import warnings

import numpy as np

warnings.filterwarnings("ignore", message="Pandas requires version")

import pandas as pd

from live_auction import (
    active_lineup_open_count,
    max_legal_bid_for_player,
    player_active_eligible,
    player_fills_open_active_slot,
    roster_size,
)


BID_ACTIONS = [
    ("conservative", 0.75),
    ("value", 0.95),
    ("fair", 1.05),
    ("aggressive", 1.18),
    ("scarcity_push", 1.35),
]


class LinUCBBidPolicy:
    def __init__(self, actions=None, alpha=0.8, feature_dim=13, seed=None):
        self.actions = actions or BID_ACTIONS
        self.alpha = alpha
        self.feature_dim = feature_dim
        self.rng = random.Random(seed)
        self.a = [np.eye(feature_dim) for _ in self.actions]
        self.b = [np.zeros(feature_dim) for _ in self.actions]
        self.observations = []

    def context_vector(self, room, team_idx, player):
        agent = room.agents[team_idx]
        positions = {str(pos).upper() for pos in player.get("positions", [])}
        value = float(player.get("dollars", 0) or 0)
        return np.array([
            1.0,
            min(value / 80.0, 1.5),
            min(agent.budget / 400.0, 1.0),
            min(max_legal_bid_for_player(agent, player) / 100.0, 1.5),
            roster_size(agent) / 40.0,
            active_lineup_open_count(agent) / 24.0,
            float(player.get("is_prospect", False)),
            float(player_active_eligible(player)),
            float(player_fills_open_active_slot(agent, player)),
            float("C" in positions),
            float("SS" in positions),
            float("SP" in positions),
            float("RP" in positions),
        ], dtype=float)

    def choose(self, context):
        scores = []
        for action_idx in range(len(self.actions)):
            a_inv = np.linalg.inv(self.a[action_idx])
            theta = a_inv @ self.b[action_idx]
            mean = float(theta @ context)
            uncertainty = float(np.sqrt(context @ a_inv @ context))
            scores.append(mean + self.alpha * uncertainty)
        action_idx = max(range(len(scores)), key=lambda idx: scores[idx])
        name, multiplier = self.actions[action_idx]
        return {
            "action_idx": action_idx,
            "action": name,
            "multiplier": multiplier,
            "score": scores[action_idx],
            "context": context,
        }

    def observe(self, decision, reward, metadata=None):
        action_idx = decision["action_idx"]
        context = decision["context"]
        self.a[action_idx] += np.outer(context, context)
        self.b[action_idx] += reward * context
        record = {
            "Action": decision["action"],
            "Multiplier": decision["multiplier"],
            "Reward": reward,
        }
        if metadata:
            record.update(metadata)
        self.observations.append(record)

    def reward_for_purchase(self, player, price, roster_row, agent):
        value = float(player.get("dollars", 0) or 0)
        surplus = value - float(price)
        slot = roster_row.get("Slot", "Bench")
        reward = surplus / 10.0
        if slot != "Bench":
            reward += 1.5
        if player.get("is_prospect", False) and not player_active_eligible(player):
            reward -= 1.0 if active_lineup_open_count(agent) > 0 else 0.2
        reward -= active_lineup_open_count(agent) * 0.08
        return float(reward)

    def action_summary(self):
        if not self.observations:
            return pd.DataFrame(columns=["Action", "Decisions", "Avg_Reward", "Avg_Multiplier"])
        df = pd.DataFrame(self.observations)
        return (
            df.groupby("Action")
            .agg(
                Decisions=("Action", "count"),
                Avg_Reward=("Reward", "mean"),
                Avg_Multiplier=("Multiplier", "mean"),
            )
            .reset_index()
            .sort_values("Avg_Reward", ascending=False)
        )
