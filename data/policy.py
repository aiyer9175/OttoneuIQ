DEFAULT_POLICY = {
    "DEFAULT": {"bid_multiplier": 1.00, "nomination_multiplier": 1.00, "max_overpay": 8},
    "SP": {"bid_multiplier": 1.08, "nomination_multiplier": 1.12, "max_overpay": 10},
    "RP": {"bid_multiplier": 0.92, "nomination_multiplier": 0.95, "max_overpay": 4},
    "C": {"bid_multiplier": 1.10, "nomination_multiplier": 1.08, "max_overpay": 7},
    "MI": {"bid_multiplier": 1.04, "nomination_multiplier": 1.05, "max_overpay": 7},
    "CI": {"bid_multiplier": 1.00, "nomination_multiplier": 1.00, "max_overpay": 6},
    "OF": {"bid_multiplier": 0.98, "nomination_multiplier": 0.98, "max_overpay": 6},
    "UTIL": {"bid_multiplier": 0.92, "nomination_multiplier": 0.90, "max_overpay": 4},
    "PROSPECT": {"bid_multiplier": 0.92, "nomination_multiplier": 0.82, "max_overpay": 3},
    "ELITE_PROSPECT": {"bid_multiplier": 1.08, "nomination_multiplier": 1.05, "max_overpay": 8},
    "STAR": {"bid_multiplier": 1.04, "nomination_multiplier": 1.20, "max_overpay": 12},
    "ENDGAME": {"bid_multiplier": 1.00, "nomination_multiplier": 0.80, "max_overpay": 1},
}


def player_policy_groups(player):
    groups = []
    positions = {str(pos).upper() for pos in player.get("positions", [])}
    value = float(player.get("dollars", 0))

    if "SP" in positions or "P" in positions:
        groups.append("SP")
    if "RP" in positions:
        groups.append("RP")
    if "C" in positions:
        groups.append("C")
    if positions.intersection({"2B", "SS", "MI"}):
        groups.append("MI")
    if positions.intersection({"1B", "3B", "CI"}):
        groups.append("CI")
    if "OF" in positions:
        groups.append("OF")
    if positions.issubset({"DH", "UTIL"}):
        groups.append("UTIL")
    if player.get("is_prospect", False):
        groups.append("ELITE_PROSPECT" if value >= 15 else "PROSPECT")
    if value >= 40:
        groups.append("STAR")
    if value <= 2:
        groups.append("ENDGAME")

    return groups or ["DEFAULT"]


def policy_average(policy, groups, key):
    values = [policy.get(group, policy["DEFAULT"]).get(key, policy["DEFAULT"][key]) for group in groups]
    return sum(values) / len(values)


def policy_multiplier(player, policy=None, key="bid_multiplier"):
    active_policy = policy or DEFAULT_POLICY
    return policy_average(active_policy, player_policy_groups(player), key)


def policy_max_overpay(player, policy=None):
    active_policy = policy or DEFAULT_POLICY
    return policy_average(active_policy, player_policy_groups(player), "max_overpay")
