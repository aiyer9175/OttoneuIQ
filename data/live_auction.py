import argparse
import os
import random
import select
import sys
import time
import warnings

warnings.filterwarnings("ignore", message="Pandas requires version")

import pandas as pd

from agent import OttoneuAgent
from policy import DEFAULT_POLICY, policy_max_overpay, policy_multiplier
from valuation import (
    ROSTER_SLOTS,
    TOTAL_ROSTER_LIMIT,
    assign_player_to_roster_slot,
    eligible_roster_slots,
    load_player_pool,
)


DEFAULT_PROSPECT_FILE = "Baseball Composite Prospect List 2026 - List.csv"
PREMIUM_POSITIONS = {"C", "SS", "SP"}
SECONDARY_POSITIONS = {"2B", "3B", "MI", "CI", "OF", "RP"}
ACTIVE_SLOT_RESERVE_DOLLARS = 3
VALID_POSITIONS = {
    "C", "1B", "2B", "3B", "SS", "MI", "CI", "OF", "UTIL", "DH",
    "SP", "RP", "P", "BENCH",
}
RECOMMENDATION_MODES = {"AUTO", "EARLY", "MID", "LATE", "PROSPECTS", "ALL"}
ROSTER_DISPLAY_ORDER = {
    "C": 0,
    "1B": 1,
    "2B": 2,
    "3B": 3,
    "SS": 4,
    "MI": 5,
    "CI": 6,
    "OF": 7,
    "UTIL": 8,
    "SP": 9,
    "RP": 10,
    "Bench": 11,
}
EARLY_MIN_VALUES = {
    "SP": 25,
    "SS": 22,
    "C": 14,
    "OF": 25,
    "1B": 20,
    "2B": 18,
    "3B": 18,
    "MI": 18,
    "CI": 18,
    "RP": 10,
    "UTIL": 25,
    "DH": 25,
    "P": 16,
    "BENCH": 25,
}


def roster_size(agent):
    return sum(agent.roster_counts.values())


def active_slot_count(agent):
    return sum(
        count for slot, count in agent.roster_counts.items()
        if slot != "Bench" and ROSTER_SLOTS.get(slot, 0) > 0
    )


def active_slots_required():
    return sum(count for slot, count in ROSTER_SLOTS.items() if slot != "Bench" and count > 0)


def active_lineup_open_count(agent):
    return active_slots_required() - active_slot_count(agent)


def player_active_eligible(player):
    return bool(player.get("active_eligible", not player.get("is_prospect", False)))


def roster_sort_key(row):
    return (
        ROSTER_DISPLAY_ORDER.get(row.get("Slot"), 99),
        -float(row.get("Salary", 0) or 0),
        str(row.get("Player", "")),
    )


def active_lineup_rows(roster_rows):
    assigned = {}
    for row in sorted(roster_rows, key=roster_sort_key):
        slot = row.get("Slot")
        if slot in {"Bench", "P"}:
            continue
        assigned.setdefault(slot, []).append(row)

    lineup = []
    active_slots = [
        slot for slot, _ in sorted(ROSTER_DISPLAY_ORDER.items(), key=lambda item: item[1])
        if slot not in {"Bench", "P"} and ROSTER_SLOTS.get(slot, 0) > 0
    ]
    for slot in active_slots:
        slot_count = ROSTER_SLOTS[slot]
        players = assigned.get(slot, [])
        for index in range(slot_count):
            row = players[index] if index < len(players) else {}
            label = slot if slot_count == 1 else f"{slot}{index + 1}"
            lineup.append({
                "Position": label,
                "Player": row.get("Player", ""),
                "Salary": row.get("Salary", ""),
                "Value": row.get("Value", ""),
                "Surplus": row.get("Surplus", ""),
                "Eligibility": row.get("Positions", ""),
            })
    return lineup


def max_legal_bid(agent):
    spots_left = TOTAL_ROSTER_LIMIT - len(agent.roster)
    if spots_left <= 0:
        return 0
    return max(agent.budget - (spots_left - 1), 0)


def player_fills_open_active_slot(agent, player):
    if not player_active_eligible(player):
        return False
    for slot in eligible_roster_slots(player["positions"]):
        if agent.roster_counts.get(slot, 0) < ROSTER_SLOTS.get(slot, 0):
            return True
    return False


def active_lineup_reserve_after_purchase(agent, player):
    open_active = active_lineup_open_count(agent)
    if player_fills_open_active_slot(agent, player):
        open_active = max(open_active - 1, 0)
    if open_active <= 0:
        return 0
    reserve_per_slot = min(
        ACTIVE_SLOT_RESERVE_DOLLARS,
        max(1, int(agent.budget / max(active_lineup_open_count(agent), 1))),
    )
    return open_active * reserve_per_slot


def max_legal_bid_for_player(agent, player):
    if roster_size(agent) >= TOTAL_ROSTER_LIMIT:
        return 0
    spots_left_after_purchase = TOTAL_ROSTER_LIMIT - roster_size(agent) - 1
    roster_reserve = max(spots_left_after_purchase, 0)
    active_reserve = active_lineup_reserve_after_purchase(agent, player)
    return max(agent.budget - max(roster_reserve, active_reserve), 0)


def scarcity_multiplier(agent, player):
    slots = eligible_roster_slots(player["positions"])
    if player_fills_open_active_slot(agent, player):
        if "C" in slots and agent.roster_counts.get("C", 0) < ROSTER_SLOTS["C"]:
            return 1.55
        if "SS" in slots and agent.roster_counts.get("SS", 0) < ROSTER_SLOTS["SS"]:
            return 1.35
        if "SP" in slots and agent.roster_counts.get("SP", 0) < ROSTER_SLOTS["SP"]:
            return 1.25
    for slot in slots:
        if slot in PREMIUM_POSITIONS and agent.roster_counts.get(slot, 0) < ROSTER_SLOTS.get(slot, 0):
            return 1.2
    for slot in slots:
        if slot in SECONDARY_POSITIONS and agent.roster_counts.get(slot, 0) < ROSTER_SLOTS.get(slot, 0):
            return 1.08
    if "UTIL" in slots and agent.roster_counts.get("UTIL", 0) < ROSTER_SLOTS.get("UTIL", 0):
        return 1.02
    return 0.82


def assign_roster_slot(agent, player):
    if not player_active_eligible(player):
        if agent.roster_counts.get("Bench", 0) < ROSTER_SLOTS["Bench"]:
            agent.roster_counts["Bench"] += 1
            return "Bench"
        return None
    return assign_player_to_roster_slot(agent.roster_counts, player["positions"])


def find_player(pool, query):
    normalized_query = query.strip().lower()
    if not normalized_query:
        return None

    exact = [p for p in pool if p["Name"].lower() == normalized_query]
    if exact:
        return exact[0]

    partial = [p for p in pool if normalized_query in p["Name"].lower()]
    if len(partial) == 1:
        return partial[0]
    if partial:
        print("Multiple matches:")
        for player in partial[:8]:
            print(f"  {player['Name']} (${player['dollars']:.2f}, {'/'.join(player['positions'])})")
    return None


class LiveAuctionRoom:
    def __init__(
        self,
        pool,
        human_team=0,
        bid_seconds=60,
        auto_human=False,
        max_picks=None,
        policy=None,
        bid_policy=None,
    ):
        self.pool = list(pool)
        self.human_team = human_team
        self.bid_seconds = bid_seconds
        self.auto_human = auto_human
        self.max_picks = max_picks
        self.policy = policy or DEFAULT_POLICY
        self.bid_policy = bid_policy
        self.agents = [OttoneuAgent(manager_id=i) for i in range(12)]
        self.nomination_cursor = 0
        self.history_rows = []
        self.roster_rows = []
        self.paused = False
        self.active_nominator_idx = None
        self.skip_current_nomination = False
        self.in_live_auction = False
        self.pending_bid_decisions = {}
        self.bandit_decision_rows = []

    def team_label(self, index):
        return f"Team {index + 1}"

    def parse_team_arg(self, arg):
        clean_arg = arg.strip().lower()
        if clean_arg in {"mine", "me", "my"}:
            return self.human_team
        if clean_arg.startswith("team "):
            clean_arg = clean_arg.replace("team ", "", 1).strip()
        if clean_arg.isdigit():
            team_num = int(clean_arg)
            if 1 <= team_num <= 12:
                return team_num - 1
        return None

    def print_commands(self):
        print("Commands:")
        print("  help            Show these commands")
        print("  rosters         Show every team's roster size, budget, and max bid")
        print("  roster mine     Show your roster")
        print("  roster 7        Show Team 7's players")
        print("  sold SP         Show sold players eligible at SP")
        print("  sold OF         Show sold players eligible at OF")
        print("  history         Show the last 10 completed picks")
        print("  history 25      Show the last 25 completed picks")
        print("  recommend       Show your best nomination targets")
        print("  recommend 10    Show your top 10 nomination targets")
        print("  recommend SP    Show recommendations filtered to SP")
        print("  recommend late  Show surplus/spec targets without early gates")
        print("  prospects       Show prospect nomination targets")
        print("  sim 60          Simulate 60 picks with AI controlling all teams")
        print("  sim end         Simulate until the auction is complete")
        print("  pause           Pause before the next nomination or pause the bid clock")
        print("  resume          Resume a paused auction")
        print("  targets SS      Show nomination targets for one position")
        print("  top             During nomination, show top available players")

    def print_roster_summary(self):
        print("\nTeam Summary")
        for idx, agent in enumerate(self.agents):
            marker = " <- you" if idx == self.human_team else ""
            print(
                f"{self.team_label(idx):>7}: "
                f"{roster_size(agent):>2}/{TOTAL_ROSTER_LIMIT} players | "
                f"budget ${agent.budget:>3} | max bid ${max_legal_bid(agent):>3}{marker}"
            )

    def print_team_roster(self, team_idx):
        rows = [row for row in self.roster_rows if row["Team"] == self.team_label(team_idx)]
        agent = self.agents[team_idx]
        print(
            f"\n{self.team_label(team_idx)} roster: "
            f"{roster_size(agent)}/{TOTAL_ROSTER_LIMIT} players | "
            f"budget ${agent.budget} | max bid ${max_legal_bid(agent)}"
        )
        if not rows:
            print("  No players bought yet.")
            return

        rows = sorted(rows, key=roster_sort_key)
        for row in rows:
            print(
                f"  {row['Slot']:<5} {row['Player']:<28} "
                f"${row['Salary']:<3} value ${row['Value']:<6.2f} "
                f"{row['Positions']}"
            )

    def print_sold_by_position(self, position):
        pos = position.strip().upper()
        if pos not in VALID_POSITIONS:
            print(f"Unknown position '{position}'. Try one of: {', '.join(sorted(VALID_POSITIONS))}")
            return

        rows = [
            row for row in self.roster_rows
            if row["Slot"].upper() == pos or pos in {p.upper() for p in row["Positions"].split("/")}
        ]
        print(f"\nSold players matching {pos}: {len(rows)}")
        if not rows:
            return

        rows = sorted(rows, key=lambda row: (-row["Salary"], row["Player"]))
        for row in rows:
            print(
                f"  {row['Player']:<28} {row['Team']:<7} "
                f"${row['Salary']:<3} slot {row['Slot']:<5} "
                f"value ${row['Value']:<6.2f} {row['Positions']}"
            )

    def handle_command(self, raw_input):
        command = raw_input.strip()
        if command.startswith(":"):
            command = command[1:].strip()
        parts = command.split()
        if not parts:
            return False

        verb = parts[0].lower()
        if verb == "help":
            self.print_commands()
            return True
        if verb == "rosters":
            self.print_roster_summary()
            return True
        if verb == "roster":
            if len(parts) < 2:
                print("Usage: roster mine OR roster 7")
                return True
            team_idx = self.parse_team_arg(" ".join(parts[1:]))
            if team_idx is None:
                print("Unknown team. Use roster mine or roster 1 through roster 12.")
                return True
            self.print_team_roster(team_idx)
            return True
        if verb == "sold":
            if len(parts) < 2:
                print("Usage: sold SP")
                return True
            self.print_sold_by_position(parts[1])
            return True
        if verb == "history":
            limit = 10
            if len(parts) >= 2 and parts[1].isdigit():
                limit = max(1, min(int(parts[1]), 200))
            self.print_history(limit=limit)
            return True
        if verb == "recommend":
            position, limit, mode = self.parse_recommendation_args(parts[1:])
            self.print_recommendations(position=position, limit=limit, mode=mode)
            return True
        if verb == "prospects":
            position, limit, _ = self.parse_recommendation_args(parts[1:])
            self.print_recommendations(position=position, limit=limit, mode="PROSPECTS")
            return True
        if verb == "targets":
            if len(parts) < 2:
                print("Usage: targets SS")
                return True
            position, limit, mode = self.parse_recommendation_args(parts[1:])
            self.print_recommendations(position=position, limit=limit, mode=mode)
            return True
        if verb in {"sim", "simulate"}:
            if len(parts) < 2:
                print("Usage: sim 60 OR sim end")
                return True
            if self.in_live_auction:
                print("Finish the current player auction before simulating future picks.")
                return True
            if parts[1].lower() in {"end", "all", "complete", "finish"}:
                self.simulate_to_end()
            elif parts[1].isdigit():
                self.simulate_picks(max(1, int(parts[1])))
            else:
                print("Usage: sim 60 OR sim end")
            return True
        if verb == "pause":
            self.paused = True
            print("Auction paused. Type 'resume' or 'play' to continue.")
            return True
        if verb in {"resume", "play"}:
            self.paused = False
            print("Auction resumed.")
            return True
        return False

    def print_history(self, limit=10):
        print(f"\nLast {min(limit, len(self.history_rows))} picks:")
        if not self.history_rows:
            print("  No completed picks yet.")
            return

        for row in self.history_rows[-limit:]:
            print(
                f"  #{row['Pick']:<3} {row['Player']:<28} "
                f"{row['Winner']:<7} ${row['Price']:<3} "
                f"slot {row['Slot']:<5} nom {row['Nominator']}"
            )

    def parse_recommendation_args(self, args):
        position = None
        limit = 8
        mode = "AUTO"
        for arg in args:
            clean_arg = arg.strip().upper()
            if clean_arg.isdigit():
                limit = max(1, min(int(clean_arg), 25))
            elif clean_arg in VALID_POSITIONS:
                position = clean_arg
            elif clean_arg in RECOMMENDATION_MODES:
                mode = clean_arg
            else:
                print(f"Ignoring unknown recommendation argument: {arg}")
        return position, limit, mode

    def player_matches_position(self, player, position):
        pos = position.strip().upper()
        if pos not in VALID_POSITIONS:
            return False
        if pos == "BENCH":
            return True
        player_positions = {str(p).strip().upper() for p in player["positions"]}
        return pos in player_positions or pos in eligible_roster_slots(player["positions"])

    def sold_rows_for_player_position(self, player=None, position=None):
        if position is not None:
            pos = position.strip().upper()
            return [
                row for row in self.roster_rows
                if row["Slot"].upper() == pos or pos in {p.upper() for p in row["Positions"].split("/")}
            ]

        if player is None:
            return list(self.roster_rows)

        slots = set(eligible_roster_slots(player["positions"]))
        positions = {str(p).strip().upper() for p in player["positions"]}
        return [
            row for row in self.roster_rows
            if row["Slot"].upper() in slots
            or positions.intersection({p.upper() for p in row["Positions"].split("/")})
        ]

    def observed_inflation(self, player=None, position=None):
        rows = self.sold_rows_for_player_position(player=player, position=position)
        usable_rows = [row for row in rows if float(row["Value"]) > 0]
        if not usable_rows:
            return 1.0

        ratios = [float(row["Salary"]) / float(row["Value"]) for row in usable_rows[-24:]]
        return max(0.55, min(sum(ratios) / len(ratios), 1.65))

    def team_has_open_starting_fit(self, team_idx, player):
        if not player_active_eligible(player):
            return False
        counts = self.agents[team_idx].roster_counts
        for slot in eligible_roster_slots(player["positions"]):
            if counts.get(slot, 0) < ROSTER_SLOTS.get(slot, 0):
                return True
        return False

    def room_context_for_player(self, player):
        value = max(float(player["dollars"]), 1.0)
        demand_teams = 0
        purse_teams = 0
        high_purse_teams = 0
        max_room_bid = 0

        for idx, agent in enumerate(self.agents):
            if idx == self.human_team:
                continue
            if not self.team_can_roster(idx, player):
                continue

            team_max_bid = max_legal_bid_for_player(agent, player)
            max_room_bid = max(max_room_bid, team_max_bid)
            if self.team_has_open_starting_fit(idx, player):
                demand_teams += 1
            if team_max_bid >= value * 0.75:
                purse_teams += 1
            if team_max_bid >= value:
                high_purse_teams += 1

        return {
            "demand_teams": demand_teams,
            "purse_teams": purse_teams,
            "high_purse_teams": high_purse_teams,
            "max_room_bid": max_room_bid,
        }

    def roster_need_bonus(self, player):
        human = self.agents[self.human_team]
        slots = eligible_roster_slots(player["positions"])
        if not self.team_can_roster(self.human_team, player):
            return -100
        if not player_active_eligible(player):
            return -8 if active_lineup_open_count(human) > 0 else -1

        for slot in slots:
            if slot in PREMIUM_POSITIONS and human.roster_counts.get(slot, 0) < ROSTER_SLOTS.get(slot, 0):
                return 7
        for slot in slots:
            if slot in SECONDARY_POSITIONS and human.roster_counts.get(slot, 0) < ROSTER_SLOTS.get(slot, 0):
                return 4
        if "UTIL" in slots and human.roster_counts.get("UTIL", 0) < ROSTER_SLOTS.get("UTIL", 0):
            return 2
        return -2

    def estimate_market_price(self, player):
        value = max(float(player["dollars"]), 1.0)
        context = self.room_context_for_player(player)
        inflation = self.observed_inflation(player=player)

        demand_pressure = 0.78 + min(context["demand_teams"], 7) * 0.045
        purse_pressure = 0.82 + min(context["purse_teams"], 7) * 0.04
        if context["high_purse_teams"] <= 2:
            purse_pressure -= 0.12
        if context["demand_teams"] <= 2:
            demand_pressure -= 0.08

        expected = value * inflation * demand_pressure * purse_pressure
        expected = max(1, min(expected, context["max_room_bid"] + 1 if context["max_room_bid"] else 1))
        return round(expected, 2), context

    def score_nomination_candidate(self, player):
        value = round(float(player["dollars"]), 2)
        human_max_bid = max_legal_bid_for_player(self.agents[self.human_team], player)
        if human_max_bid < 1 or not self.team_can_roster(self.human_team, player):
            return None

        expected_price, context = self.estimate_market_price(player)
        expected_buy_price = min(expected_price, human_max_bid)
        expected_surplus = round(value - expected_buy_price, 2)
        need_bonus = self.roster_need_bonus(player)

        bargain_window_bonus = 0
        if context["high_purse_teams"] <= 2:
            bargain_window_bonus += 3
        if context["purse_teams"] <= 4:
            bargain_window_bonus += 2

        room_filled_bonus = 0
        if context["demand_teams"] <= 3:
            room_filled_bonus += 2

        prospect_bonus = 2 if player.get("is_prospect", False) and expected_surplus > -2 else 0
        active_open = active_lineup_open_count(self.agents[self.human_team])
        prospect_roster_penalty = 0
        if player.get("is_prospect", False) and not player_active_eligible(player):
            prospect_bonus = 0
            prospect_roster_penalty = 12 if active_open > 0 else 3
        bidding_war_risk = max(context["demand_teams"] - 5, 0) * 1.5
        affordability_penalty = 5 if value > human_max_bid * 1.2 else 0
        bench_penalty = 3 if need_bonus < 0 and expected_surplus < 4 else 0
        low_value_penalty = max(0, 8 - value) * 1.1
        early_market_bonus = self.early_market_anchor_bonus(value, context)

        score = (
            expected_surplus
            + need_bonus
            + bargain_window_bonus
            + room_filled_bonus
            + prospect_bonus
            + early_market_bonus
            - bidding_war_risk
            - affordability_penalty
            - bench_penalty
            - prospect_roster_penalty
            - low_value_penalty
        )

        intent = "BUY" if expected_surplus > 0 else "WAIT"
        if expected_surplus < 0 and context["demand_teams"] >= 5 and value >= 8:
            intent = "DRAIN"
        elif need_bonus < 0 and expected_surplus < 3:
            intent = "WAIT"
        elif value < 3 and expected_surplus < 2:
            intent = "WAIT"
        elif prospect_roster_penalty and expected_surplus < prospect_roster_penalty:
            intent = "WAIT"

        reason_parts = []
        if expected_surplus >= 3:
            reason_parts.append("surplus")
        if early_market_bonus >= 5:
            reason_parts.append("early anchor")
        if need_bonus >= 4:
            reason_parts.append("fills need")
        if context["high_purse_teams"] <= 2:
            reason_parts.append("limited purses")
        if context["demand_teams"] <= 3:
            reason_parts.append("room filled")
        if player.get("is_prospect", False):
            reason_parts.append("prospect")
        if not reason_parts:
            reason_parts.append("market fit")

        return {
            "Player": player["Name"],
            "Positions": "/".join(player["positions"]),
            "Value": value,
            "Expected_Price": expected_price,
            "Expected_Surplus": expected_surplus,
            "Score": round(score, 2),
            "Intent": intent,
            "Demand_Teams": context["demand_teams"],
            "High_Purse_Teams": context["high_purse_teams"],
            "Reason": ", ".join(reason_parts),
            "Is_Prospect": bool(player.get("is_prospect", False)),
        }

    def early_market_anchor_bonus(self, value, context):
        picks_completed = len(self.history_rows)
        if picks_completed >= 48 or context["demand_teams"] < 5:
            return 0

        phase_weight = 1 - (picks_completed / 48)
        nomination_weight = min(value, 70) * 0.35
        return nomination_weight * phase_weight

    def recommendation_phase(self):
        picks_completed = len(self.history_rows)
        average_budget = sum(agent.budget for agent in self.agents) / len(self.agents)
        if picks_completed < 60 and average_budget > 300:
            return "EARLY"
        if picks_completed > 300 or average_budget < 120:
            return "LATE"
        return "MID"

    def early_min_value_for_player(self, player, position=None):
        slots = eligible_roster_slots(player["positions"])
        if position:
            return EARLY_MIN_VALUES.get(position.upper(), 20)
        if not slots:
            return 20
        return min(EARLY_MIN_VALUES.get(slot, 20) for slot in slots)

    def passes_recommendation_gate(self, row, player, mode, position=None):
        effective_mode = self.recommendation_phase() if mode == "AUTO" else mode
        value = row["Value"]

        if effective_mode == "ALL":
            return True

        if effective_mode == "PROSPECTS":
            return row["Is_Prospect"]

        if effective_mode == "EARLY":
            min_value = self.early_min_value_for_player(player, position=position)
            if row["Is_Prospect"] and value < max(min_value, 20):
                return False
            if value < min_value:
                return False
            return row["Intent"] in {"BUY", "DRAIN"} or row["Score"] >= 0

        if effective_mode == "MID":
            if row["Is_Prospect"] and value < 8 and row["Expected_Surplus"] < 3:
                return False
            return row["Intent"] != "WAIT" or row["Expected_Surplus"] >= 2

        if effective_mode == "LATE":
            return row["Intent"] in {"BUY", "WAIT"} or row["Expected_Surplus"] >= 0

        return True

    def recommendation_rows(self, position=None, limit=8, mode="AUTO"):
        rows = []
        for player in self.pool:
            if position and not self.player_matches_position(player, position):
                continue
            row = self.score_nomination_candidate(player)
            if row is not None and self.passes_recommendation_gate(row, player, mode, position=position):
                rows.append(row)

        effective_mode = self.recommendation_phase() if mode == "AUTO" else mode
        rows.sort(
            key=lambda row: (
                self.intent_rank(row["Intent"], effective_mode),
                -row["Score"],
                -row["Expected_Surplus"],
                -row["Value"],
            )
        )
        return rows[:limit]

    def intent_rank(self, intent, mode):
        if mode == "EARLY":
            return {"DRAIN": 0, "BUY": 1, "WAIT": 2}.get(intent, 3)
        if mode in {"LATE", "PROSPECTS"}:
            return {"BUY": 0, "WAIT": 1, "DRAIN": 2}.get(intent, 3)
        return {"BUY": 0, "DRAIN": 1, "WAIT": 2}.get(intent, 3)

    def print_recommendations(self, position=None, limit=8, mode="AUTO"):
        pos_text = f" for {position.upper()}" if position else ""
        effective_mode = self.recommendation_phase() if mode == "AUTO" else mode
        rows = self.recommendation_rows(position=position, limit=limit, mode=mode)
        print(f"\nRecommended nominations{pos_text} [{effective_mode.lower()}]:")
        if not rows:
            print("  No legal targets found.")
            return

        for idx, row in enumerate(rows, start=1):
            print(
                f"{idx:>2}. {row['Player']:<28} {row['Positions']:<10} "
                f"{row['Intent']:<5} score {row['Score']:>6.2f}"
            )
            print(
                f"    value ${row['Value']:.2f} | expected ${row['Expected_Price']:.2f} | "
                f"surplus {row['Expected_Surplus']:+.2f} | "
                f"demand {row['Demand_Teams']} | rich bidders {row['High_Purse_Teams']} | "
                f"{row['Reason']}"
            )

    def team_can_roster(self, team_idx, player):
        agent = self.agents[team_idx]
        if max_legal_bid_for_player(agent, player) < 1 or roster_size(agent) >= TOTAL_ROSTER_LIMIT:
            return False
        if not player_active_eligible(player):
            return agent.roster_counts.get("Bench", 0) < ROSTER_SLOTS["Bench"]
        return agent.has_open_slot(player["positions"])

    def remove_player(self, player):
        self.pool = [p for p in self.pool if p["Name"] != player["Name"]]

    def ai_limit(self, team_idx, player, record_decision=False):
        agent = self.agents[team_idx]
        if not self.team_can_roster(team_idx, player):
            return 0

        inflation = 1.08 * scarcity_multiplier(agent, player)
        inflation *= policy_multiplier(player, self.policy, key="bid_multiplier")
        if player.get("is_prospect", False):
            inflation *= 1.1
        if player.get("is_prospect", False) and not player_active_eligible(player):
            if active_lineup_open_count(agent) > 0:
                inflation *= 0.45
            else:
                inflation *= 0.75
        if self.bid_policy is not None:
            context = self.bid_policy.context_vector(self, team_idx, player)
            decision = self.bid_policy.choose(context)
            inflation *= decision["multiplier"]
            if record_decision:
                self.pending_bid_decisions[(player["Name"], team_idx)] = decision

        limit = agent.get_limit_price(player["dollars"], inflation)
        max_policy_bid = int(player["dollars"] + policy_max_overpay(player, self.policy))
        limit = min(limit, max_policy_bid)
        legal_max_bid = max_legal_bid_for_player(agent, player)
        if legal_max_bid >= 1 and player["dollars"] > 0:
            return max(min(limit, legal_max_bid), 1)
        return max(min(limit, legal_max_bid), 0)

    def nomination_score(self, team_idx, player):
        agent = self.agents[team_idx]
        if not self.team_can_roster(team_idx, player):
            return -1

        max_bid = max_legal_bid_for_player(agent, player)
        target_price = max(player["dollars"], 1)
        if target_price > max_bid * 1.35:
            return -1

        score = target_price * scarcity_multiplier(agent, player)
        score *= policy_multiplier(player, self.policy, key="nomination_multiplier")
        if player.get("is_prospect", False):
            score *= 1.12
        if player.get("is_prospect", False) and not player_active_eligible(player):
            score *= 0.25 if active_lineup_open_count(agent) > 0 else 0.55
        score += random.random() * 2.5
        return score

    def ai_nominate(self, team_idx):
        candidates = sorted(
            self.pool,
            key=lambda player: self.nomination_score(team_idx, player),
            reverse=True,
        )
        for player in candidates:
            if self.nomination_score(team_idx, player) > 0 and self.ai_limit(team_idx, player) >= 1:
                return player
        legal_candidates = [player for player in self.pool if self.team_can_roster(team_idx, player)]
        if legal_candidates:
            bid_candidates = [player for player in legal_candidates if self.ai_limit(team_idx, player) >= 1]
            if bid_candidates:
                return min(bid_candidates, key=lambda player: player["dollars"])
            return min(legal_candidates, key=lambda player: player["dollars"])
        return None

    def human_nominate(self):
        suggested = self.ai_nominate(self.human_team)
        print("\nYour nomination turn.")
        if suggested:
            print(f"Press Enter to nominate suggested player: {suggested['Name']} (${suggested['dollars']:.2f})")
        print("Or type a player name. Type 'recommend', 'top', or 'help'.")

        while True:
            choice = input("Nomination> ").strip()
            if not choice and suggested:
                return suggested
            if self.handle_command(choice):
                if self.paused:
                    self.paused_prompt()
                if self.skip_current_nomination:
                    self.skip_current_nomination = False
                    return None
                suggested = self.ai_nominate(self.human_team)
                continue
            if choice.lower() == "top":
                for player in self.pool[:10]:
                    print(f"  {player['Name']} - ${player['dollars']:.2f} - {'/'.join(player['positions'])}")
                continue
            player = find_player(self.pool, choice)
            if player and self.team_can_roster(self.human_team, player):
                return player
            print("Choose an available player who fits your roster and budget.")

    def next_nominator(self):
        for offset in range(12):
            team_idx = (self.nomination_cursor + offset) % 12
            if roster_size(self.agents[team_idx]) < TOTAL_ROSTER_LIMIT:
                self.nomination_cursor = (team_idx + 1) % 12
                return team_idx
        return None

    def nominate_player(self, team_idx):
        if team_idx == self.human_team and not self.auto_human:
            self.active_nominator_idx = team_idx
            player = self.human_nominate()
            self.active_nominator_idx = None
        else:
            player = self.ai_nominate(team_idx)

        if player is None:
            return None

        self.remove_player(player)
        print(f"\n{self.team_label(team_idx)} nominates {player['Name']} ({'/'.join(player['positions'])})")
        print(f"Model value: ${player['dollars']:.2f} | Prospect: {bool(player.get('is_prospect', False))}")
        return player

    def simulated_nominate_player(self, team_idx):
        player = self.ai_nominate(team_idx)
        if player is None:
            return None
        self.remove_player(player)
        return player

    def human_bid_prompt(self, current_bid, high_bidder, human_passed):
        if human_passed or high_bidder == self.human_team:
            return None
        max_bid = max_legal_bid(self.agents[self.human_team])
        next_bid = current_bid + 1
        print(
            f"Your max legal bid: ${max_bid}. Type bid amount, 'b' for ${next_bid}, "
            "pass, recommend, roster mine, rosters, sold SP, or help."
        )
        return "ready"

    def read_human_input(self, seconds=1):
        ready, _, _ = select.select([sys.stdin], [], [], seconds)
        if ready:
            return sys.stdin.readline().strip()
        return None

    def run_auction_for_player(self, player, nominator_idx):
        self.in_live_auction = True
        current_bid = 1
        high_bidder = nominator_idx
        passed = set()
        ai_limits = {i: self.ai_limit(i, player) for i in range(12) if i != self.human_team}

        if not self.team_can_roster(nominator_idx, player):
            high_bidder = None
            current_bid = 0

        print(f"Opening bid: ${max(current_bid, 1)} by {self.team_label(nominator_idx)}")
        deadline = time.monotonic() + self.bid_seconds
        last_ai_action = 0
        last_status_remaining = None
        last_prompt_state = None
        human_passed = False

        while time.monotonic() < deadline:
            remaining = max(int(deadline - time.monotonic()), 0)
            if remaining != last_status_remaining and remaining % 10 == 0:
                last_status_remaining = remaining
                leader = self.team_label(high_bidder) if high_bidder is not None else "No bids"
                print(f"[{remaining:02d}s] High bid: ${current_bid} by {leader}")

            if not self.auto_human:
                prompt_state = (current_bid, high_bidder, human_passed)
                if prompt_state != last_prompt_state:
                    self.human_bid_prompt(current_bid, high_bidder, human_passed)
                    last_prompt_state = prompt_state
                human_input = self.read_human_input(seconds=1)
                if human_input:
                    lower_input = human_input.lower()
                    if self.handle_command(human_input):
                        if self.paused:
                            paused_at = time.monotonic()
                            self.paused_prompt()
                            deadline += time.monotonic() - paused_at
                        last_prompt_state = None
                        continue
                    if lower_input == "pass":
                        human_passed = True
                        passed.add(self.human_team)
                        print("You passed on this player.")
                    else:
                        bid = current_bid + 1 if lower_input == "b" else None
                        if lower_input.isdigit():
                            bid = int(lower_input)
                        if bid is None:
                            print("Enter a whole-dollar bid, 'b', or 'pass'.")
                        elif bid <= current_bid:
                            print(f"Bid must beat the current ${current_bid} price.")
                        elif bid > max_legal_bid_for_player(self.agents[self.human_team], player):
                            print("That bid would violate your cap reserve.")
                        elif not self.team_can_roster(self.human_team, player):
                            print("You do not have a legal roster slot for this player.")
                        else:
                            current_bid = bid
                            high_bidder = self.human_team
                            print(f"You bid ${current_bid}.")
                else:
                    pass
            else:
                time.sleep(min(0.05, max(self.bid_seconds, 0)))

            if time.monotonic() - last_ai_action >= 1:
                last_ai_action = time.monotonic()
                bidder = self.choose_ai_bidder(ai_limits, current_bid, high_bidder, passed)
                if bidder is not None:
                    current_bid = self.next_ai_bid(ai_limits[bidder], current_bid)
                    high_bidder = bidder
                    print(f"{self.team_label(bidder)} bids ${current_bid}.")

            if self.no_live_bidders(player, ai_limits, current_bid, high_bidder, passed, human_passed):
                break

        if high_bidder is None:
            print(f"No legal bids for {player['Name']}. Returning player to pool.")
            self.pool.append(player)
            self.in_live_auction = False
            return

        self.complete_purchase(player, high_bidder, current_bid, nominator_idx)
        self.in_live_auction = False

    def simulated_auction_for_player(self, player, nominator_idx):
        limit_orders = []
        self.pending_bid_decisions = {}
        for team_idx in range(12):
            if self.team_can_roster(team_idx, player):
                bid = self.ai_limit(team_idx, player, record_decision=self.bid_policy is not None)
                if bid >= 1:
                    limit_orders.append((bid, team_idx))

        if not limit_orders:
            self.pool.append(player)
            return None

        limit_orders.sort(reverse=True, key=lambda item: item[0])
        highest_bid, winner_idx = limit_orders[0]
        if len(limit_orders) > 1:
            price = min(highest_bid, limit_orders[1][0] + 1)
        else:
            price = 1

        self.complete_purchase(player, winner_idx, price, nominator_idx, quiet=True)
        self.observe_bandit_purchase(player, winner_idx, price)
        return self.history_rows[-1]

    def observe_bandit_purchase(self, player, winner_idx, price):
        if self.bid_policy is None or not self.history_rows:
            return
        decision = self.pending_bid_decisions.get((player["Name"], winner_idx))
        if decision is None:
            return
        roster_row = self.roster_rows[-1]
        reward = self.bid_policy.reward_for_purchase(player, price, roster_row, self.agents[winner_idx])
        metadata = {
            "Pick": self.history_rows[-1]["Pick"],
            "Team": self.team_label(winner_idx),
            "Player": player["Name"],
            "Positions": "/".join(player["positions"]),
            "Price": price,
            "Value": round(float(player["dollars"]), 2),
            "Slot": roster_row["Slot"],
            "Is_Prospect": bool(player.get("is_prospect", False)),
        }
        self.bid_policy.observe(decision, reward, metadata=metadata)
        self.bandit_decision_rows.append({**metadata, "Action": decision["action"], "Reward": reward})

    def choose_ai_bidder(self, ai_limits, current_bid, high_bidder, passed):
        next_bid = current_bid + 1
        willing = [
            team_idx
            for team_idx, limit in ai_limits.items()
            if team_idx != high_bidder and team_idx not in passed and limit >= next_bid
        ]
        if not willing:
            for team_idx, limit in ai_limits.items():
                if limit < next_bid:
                    passed.add(team_idx)
            return None

        willing.sort(key=lambda team_idx: ai_limits[team_idx], reverse=True)
        return willing[0]

    def next_ai_bid(self, limit, current_bid):
        next_bid = current_bid + 1
        if limit <= next_bid:
            return next_bid
        gap = limit - current_bid
        jump = max(1, min(5, int(gap * 0.25)))
        return min(limit, current_bid + jump)

    def no_live_bidders(self, player, ai_limits, current_bid, high_bidder, passed, human_passed):
        next_bid = current_bid + 1
        ai_can_bid = any(
            team_idx != high_bidder and team_idx not in passed and limit >= next_bid
            for team_idx, limit in ai_limits.items()
        )
        human_can_bid = (
            not self.auto_human
            and not human_passed
            and high_bidder != self.human_team
            and self.team_can_roster(self.human_team, player)
            and max_legal_bid_for_player(self.agents[self.human_team], player) >= next_bid
        )
        return not ai_can_bid and not human_can_bid

    def complete_purchase(self, player, winner_idx, price, nominator_idx, quiet=False):
        winner = self.agents[winner_idx]
        slot = assign_roster_slot(winner, player)
        winner.budget -= price
        winner.roster.append(player["Name"])

        reward = (player["dollars"] - price) + 2.0
        winner.learn_from_result(reward)

        row = {
            "Pick": len(self.history_rows) + 1,
            "Nominator": self.team_label(nominator_idx),
            "Player": player["Name"],
            "Positions": "/".join(player["positions"]),
            "Value": round(player["dollars"], 2),
            "Price": price,
            "Winner": self.team_label(winner_idx),
            "Slot": slot,
            "Is_Prospect": bool(player.get("is_prospect", False)),
        }
        self.history_rows.append(row)
        self.roster_rows.append({
            "Team": self.team_label(winner_idx),
            "Player": player["Name"],
            "Slot": slot,
            "Positions": "/".join(player["positions"]),
            "Salary": price,
            "Value": round(player["dollars"], 2),
            "Surplus": round(player["dollars"] - price, 2),
            "Is_Prospect": bool(player.get("is_prospect", False)),
        })

        if not quiet:
            print(f"SOLD: {player['Name']} to {self.team_label(winner_idx)} for ${price} ({slot})")
        if not quiet and winner_idx == self.human_team:
            print(f"Your budget: ${winner.budget} | Roster: {roster_size(winner)} / {TOTAL_ROSTER_LIMIT}")

    def simulate_one_pick(self):
        if not self.pool:
            return None
        if sum(roster_size(agent) for agent in self.agents) >= 12 * TOTAL_ROSTER_LIMIT:
            return None

        nominator_idx = self.next_nominator()
        if nominator_idx is None:
            return None

        player = self.simulated_nominate_player(nominator_idx)
        if player is None:
            return None
        return self.simulated_auction_for_player(player, nominator_idx)

    def simulate_picks(self, count):
        start_pick = len(self.history_rows)
        if self.active_nominator_idx is not None and count > 0:
            player = self.simulated_nominate_player(self.active_nominator_idx)
            if player is not None:
                self.simulated_auction_for_player(player, self.active_nominator_idx)
                self.skip_current_nomination = True
                count -= 1

        for _ in range(count):
            if self.simulate_one_pick() is None:
                break

        simulated = len(self.history_rows) - start_pick
        self.save_reports()
        print(f"Simulated {simulated} picks. Current pick count: {len(self.history_rows)}.")
        if simulated:
            self.print_history(limit=min(simulated, 10))

    def simulate_to_end(self):
        remaining_slots = (12 * TOTAL_ROSTER_LIMIT) - sum(roster_size(agent) for agent in self.agents)
        self.simulate_picks(max(remaining_slots, 0))

    def paused_prompt(self):
        while self.paused:
            command = input("Paused> ").strip()
            if not command:
                continue
            self.handle_command(command)

    def save_reports(self):
        pd.DataFrame(self.history_rows).to_csv("live_auction_results.csv", index=False)
        pd.DataFrame(self.roster_rows).to_csv("live_team_rosters.csv", index=False)

    def run(self):
        print("--- Ottoneu Live Auction Room ---")
        print(f"You control {self.team_label(self.human_team)}.")
        print(f"Each player auction is timed for {self.bid_seconds} seconds.")
        print("Bid with a number, 'b' for the next dollar, or 'pass'.")
        print("Inspection commands: recommend, targets SS, rosters, history, sim 60, pause, help.")

        while self.pool:
            if self.max_picks is not None and len(self.history_rows) >= self.max_picks:
                break
            if sum(roster_size(agent) for agent in self.agents) >= 12 * TOTAL_ROSTER_LIMIT:
                break

            if self.paused:
                self.paused_prompt()
                continue

            nominator_idx = self.next_nominator()
            if nominator_idx is None:
                break

            player = self.nominate_player(nominator_idx)
            if player is None:
                continue
            self.run_auction_for_player(player, nominator_idx)
            self.save_reports()

        print("\nAuction complete or stopped.")
        print("Reports written: live_auction_results.csv and live_team_rosters.csv")


def parse_args():
    parser = argparse.ArgumentParser(description="Playable Ottoneu live auction room.")
    parser.add_argument("--human-team", type=int, default=1, choices=range(1, 13), help="Team number to control.")
    parser.add_argument("--bid-seconds", type=int, default=60, help="Seconds per player auction.")
    parser.add_argument("--max-picks", type=int, default=None, help="Optional cap for test/demo runs.")
    parser.add_argument("--auto-human", action="store_true", help="Let the human team auto-nominate/pass for smoke tests.")
    parser.add_argument("--batters", default="batters_auctioncalc.csv")
    parser.add_argument("--pitchers", default="pitchers_auctioncalc.csv")
    parser.add_argument("--prospects", default=DEFAULT_PROSPECT_FILE)
    parser.add_argument("--data-source", choices=["static", "cache"], default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed()

    if not os.path.exists(args.batters) and os.path.exists(os.path.join("data", args.batters)):
        os.chdir("data")
    if args.data_source:
        from data_sources import resolve_data_paths

        paths = resolve_data_paths(args.data_source)
        args.batters = paths.batters_auction
        args.pitchers = paths.pitchers_auction
        args.prospects = paths.prospects

    pool = load_player_pool(args.batters, args.pitchers, args.prospects)
    room = LiveAuctionRoom(
        pool=pool,
        human_team=args.human_team - 1,
        bid_seconds=args.bid_seconds,
        auto_human=args.auto_human,
        max_picks=args.max_picks,
    )
    try:
        room.run()
    except KeyboardInterrupt:
        room.save_reports()
        print("\nAuction paused. Current reports were saved.")


if __name__ == "__main__":
    main()
