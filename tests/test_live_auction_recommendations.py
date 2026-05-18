import os
import sys


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
sys.path.insert(0, os.path.abspath(DATA_DIR))

from live_auction import LiveAuctionRoom, active_lineup_rows, active_lineup_open_count, roster_sort_key
from valuation import ROSTER_SLOTS


def make_player(name, value, positions, is_prospect=False):
    return {
        "Name": name,
        "dollars": value,
        "positions": positions,
        "is_prospect": is_prospect,
        "active_eligible": not is_prospect,
    }


def test_recommendation_prefers_low_demand_surplus_fit():
    target = make_player("Bargain Shortstop", 24, ["SS"])
    expensive = make_player("Overheated Pitcher", 25, ["SP"])
    room = LiveAuctionRoom([target, expensive], human_team=0, auto_human=True)

    for idx in range(1, 12):
        room.agents[idx].roster_counts["SS"] = ROSTER_SLOTS["SS"]
        room.agents[idx].roster_counts["MI"] = ROSTER_SLOTS["MI"]
        room.agents[idx].roster_counts["UTIL"] = ROSTER_SLOTS["UTIL"]

    rows = room.recommendation_rows(limit=2, mode="ALL")

    assert rows[0]["Player"] == "Bargain Shortstop"
    assert rows[0]["Expected_Surplus"] > 0
    assert rows[0]["Demand_Teams"] == 0


def test_targets_filters_by_position_or_slot():
    ss = make_player("Shortstop Target", 18, ["SS"])
    sp = make_player("Starter Target", 18, ["SP"])
    room = LiveAuctionRoom([ss, sp], human_team=0, auto_human=True)

    rows = room.recommendation_rows(position="MI", limit=5, mode="ALL")

    assert [row["Player"] for row in rows] == ["Shortstop Target"]


def test_recommendation_blocks_unaffordable_full_human_roster():
    player = make_player("Blocked Player", 10, ["OF"])
    room = LiveAuctionRoom([player], human_team=0, auto_human=True)
    room.agents[0].roster_counts["Bench"] = ROSTER_SLOTS["Bench"]
    room.agents[0].roster_counts["OF"] = ROSTER_SLOTS["OF"]
    room.agents[0].roster_counts["UTIL"] = ROSTER_SLOTS["UTIL"]

    assert room.recommendation_rows(limit=5) == []


def test_early_recommendation_prefers_viable_anchor_over_low_dollar_name():
    anchor = make_player("Auction Anchor", 60, ["SP"])
    low_dollar = make_player("Spec Prospect", 8, ["SP"], is_prospect=True)
    room = LiveAuctionRoom([low_dollar, anchor], human_team=0, auto_human=True)

    rows = room.recommendation_rows(limit=2)

    assert rows[0]["Player"] == "Auction Anchor"
    assert "early anchor" in rows[0]["Reason"]


def test_recommend_command_accepts_position_alias():
    room = LiveAuctionRoom([], human_team=0, auto_human=True)

    assert room.parse_recommendation_args(["SP"]) == ("SP", 8, "AUTO")
    assert room.parse_recommendation_args(["SP", "12"]) == ("SP", 12, "AUTO")
    assert room.parse_recommendation_args(["10"]) == (None, 10, "AUTO")
    assert room.parse_recommendation_args(["late", "SS", "6"]) == ("SS", 6, "LATE")


def test_early_default_recommendation_gates_cheap_prospects():
    anchor = make_player("Auction Anchor", 60, ["SP"])
    prospect = make_player("Cheap Prospect", 8, ["SS"], is_prospect=True)
    room = LiveAuctionRoom([prospect, anchor], human_team=0, auto_human=True)

    default_rows = room.recommendation_rows(limit=5)
    prospect_rows = room.recommendation_rows(limit=5, mode="PROSPECTS")

    assert [row["Player"] for row in default_rows] == ["Auction Anchor"]
    assert [row["Player"] for row in prospect_rows] == ["Cheap Prospect"]


def test_recommend_sp_early_returns_high_value_starters():
    ace = make_player("Ace Starter", 50, ["SP"])
    cheap_sp = make_player("Cheap Starter", 8, ["SP"], is_prospect=True)
    hitter = make_player("Star Hitter", 55, ["OF"])
    room = LiveAuctionRoom([cheap_sp, hitter, ace], human_team=0, auto_human=True)

    rows = room.recommendation_rows(position="SP", limit=5)

    assert [row["Player"] for row in rows] == ["Ace Starter"]


def test_streamlit_style_ai_round_leaves_human_chance_to_bid():
    player = make_player("Auction Star", 60, ["SP"])
    room = LiveAuctionRoom([player], human_team=0, auto_human=False)
    current_bid = 1
    high_bidder = room.human_team
    passed = set()
    ai_limits = {i: room.ai_limit(i, player) for i in range(12) if i != room.human_team}

    while True:
        bidder = room.choose_ai_bidder(ai_limits, current_bid, high_bidder, passed)
        if bidder is None:
            break
        current_bid = room.next_ai_bid(ai_limits[bidder], current_bid)
        high_bidder = bidder
        if room.no_live_bidders(player, ai_limits, current_bid, high_bidder, passed, human_passed=False):
            break

    assert high_bidder != room.human_team
    assert current_bid > 1
    assert not room.no_live_bidders(player, ai_limits, current_bid, high_bidder, passed, human_passed=False)
    assert room.no_live_bidders(player, ai_limits, current_bid, high_bidder, passed, human_passed=True)


def test_roster_sort_order_lists_starters_before_bench():
    rows = [
        {"Slot": "Bench", "Player": "Reserve", "Salary": 20},
        {"Slot": "SP", "Player": "Starter", "Salary": 5},
        {"Slot": "C", "Player": "Catcher", "Salary": 1},
        {"Slot": "OF", "Player": "Outfielder", "Salary": 10},
    ]

    ordered = [row["Player"] for row in sorted(rows, key=roster_sort_key)]

    assert ordered == ["Catcher", "Outfielder", "Starter", "Reserve"]


def test_active_lineup_rows_demarcate_slots_and_exclude_bench():
    rows = [
        {"Slot": "Bench", "Player": "Reserve Bat", "Salary": 4, "Value": 3, "Surplus": -1, "Positions": "OF"},
        {"Slot": "C", "Player": "Primary Catcher", "Salary": 8, "Value": 9, "Surplus": 1, "Positions": "C"},
        {"Slot": "SP", "Player": "Ace Starter", "Salary": 45, "Value": 50, "Surplus": 5, "Positions": "SP"},
        {"Slot": "MI", "Player": "Middle Infielder", "Salary": 7, "Value": 8, "Surplus": 1, "Positions": "SS"},
    ]

    lineup = active_lineup_rows(rows)
    by_position = {row["Position"]: row for row in lineup}

    assert by_position["C1"]["Player"] == "Primary Catcher"
    assert by_position["C2"]["Player"] == ""
    assert by_position["MI"]["Player"] == "Middle Infielder"
    assert by_position["SP1"]["Player"] == "Ace Starter"
    assert "Reserve Bat" not in {row["Player"] for row in lineup}


def test_prospect_only_players_go_to_bench_not_active_slots():
    prospect = make_player("Low Minors Arm", 8, ["SP"], is_prospect=True)
    room = LiveAuctionRoom([prospect], human_team=0, auto_human=True)

    room.complete_purchase(prospect, winner_idx=0, price=3, nominator_idx=0, quiet=True)

    assert room.roster_rows[-1]["Slot"] == "Bench"
    assert active_lineup_open_count(room.agents[0]) > 0


def test_ai_prefers_active_mlb_need_before_low_level_prospect_stash():
    mlb_starter = make_player("Back End MLB Starter", 5, ["SP"])
    prospect = make_player("AA Upside Arm", 8, ["SP"], is_prospect=True)
    room = LiveAuctionRoom([prospect, mlb_starter], human_team=0, auto_human=True)

    assert room.ai_nominate(0)["Name"] == "Back End MLB Starter"
    assert room.ai_limit(0, prospect) < room.ai_limit(0, mlb_starter)


def test_simulate_picks_uses_all_teams_and_preserves_caps():
    players = [
        make_player("Ace Starter", 50, ["SP"]),
        make_player("Star Hitter", 45, ["OF"]),
        make_player("Shortstop", 35, ["SS"]),
        make_player("Catcher", 20, ["C"]),
    ]
    room = LiveAuctionRoom(players, human_team=0, auto_human=True)

    room.simulate_picks(3)

    assert len(room.history_rows) == 3
    assert len(room.roster_rows) == 3
    for agent in room.agents:
        assert agent.budget >= 0
        assert max_legal_or_zero(agent) >= 0


def test_history_command_accepts_numeric_limit():
    players = [
        make_player("Ace Starter", 50, ["SP"]),
        make_player("Star Hitter", 45, ["OF"]),
    ]
    room = LiveAuctionRoom(players, human_team=0, auto_human=True)

    room.simulate_picks(2)

    assert room.handle_command("history 1")


def test_sim_command_fast_forwards_requested_count():
    players = [
        make_player("Ace Starter", 50, ["SP"]),
        make_player("Star Hitter", 45, ["OF"]),
        make_player("Shortstop", 35, ["SS"]),
    ]
    room = LiveAuctionRoom(players, human_team=0, auto_human=True)

    assert room.handle_command("sim 2")
    assert len(room.history_rows) == 2


def test_sim_command_can_take_over_pending_human_nomination():
    players = [
        make_player("Ace Starter", 50, ["SP"]),
        make_player("Star Hitter", 45, ["OF"]),
        make_player("Shortstop", 35, ["SS"]),
    ]
    room = LiveAuctionRoom(players, human_team=0, auto_human=True)
    room.active_nominator_idx = 0

    room.simulate_picks(2)

    assert len(room.history_rows) == 2
    assert room.skip_current_nomination


def test_sim_command_is_rejected_during_live_bidding():
    players = [make_player("Ace Starter", 50, ["SP"])]
    room = LiveAuctionRoom(players, human_team=0, auto_human=True)
    room.in_live_auction = True

    assert room.handle_command("sim 1")
    assert len(room.history_rows) == 0


def test_sim_end_fast_forwards_until_no_more_players():
    players = [
        make_player("Ace Starter", 50, ["SP"]),
        make_player("Star Hitter", 45, ["OF"]),
        make_player("Shortstop", 35, ["SS"]),
    ]
    room = LiveAuctionRoom(players, human_team=0, auto_human=True)

    assert room.handle_command("sim end")
    assert len(room.history_rows) == 3
    assert not room.pool


def test_ai_nomination_falls_back_to_legal_low_value_player():
    player = make_player("Endgame Dollar", 5, ["OF"])
    room = LiveAuctionRoom([player], human_team=0, auto_human=True)
    room.agents[0].budget = 1
    room.agents[0].roster = [f"Player {idx}" for idx in range(39)]
    room.agents[0].roster_counts = {slot: count for slot, count in ROSTER_SLOTS.items()}
    room.agents[0].roster_counts["Bench"] = ROSTER_SLOTS["Bench"] - 1

    assert room.ai_nominate(0)["Name"] == "Endgame Dollar"


def test_ai_nomination_fallback_can_select_fractional_player_at_minimum_bid():
    zero_bid = make_player("Fractional Player", 0.2, ["OF"])
    dollar_bid = make_player("Dollar Player", 5, ["OF"])
    room = LiveAuctionRoom([zero_bid, dollar_bid], human_team=0, auto_human=True)
    room.agents[0].budget = 1
    room.agents[0].roster = [f"Player {idx}" for idx in range(39)]
    room.agents[0].roster_counts = {slot: count for slot, count in ROSTER_SLOTS.items()}
    room.agents[0].roster_counts["Bench"] = ROSTER_SLOTS["Bench"] - 1

    assert room.ai_nominate(0)["Name"] == "Fractional Player"


def test_ai_limit_floors_positive_players_to_one_dollar_when_legal():
    player = make_player("Fractional Player", 0.2, ["OF"])
    room = LiveAuctionRoom([player], human_team=0, auto_human=True)
    room.agents[0].budget = 1
    room.agents[0].roster = [f"Player {idx}" for idx in range(39)]
    room.agents[0].roster_counts = {slot: count for slot, count in ROSTER_SLOTS.items()}
    room.agents[0].roster_counts["Bench"] = ROSTER_SLOTS["Bench"] - 1

    assert room.ai_limit(0, player) == 1


def max_legal_or_zero(agent):
    roster_count = sum(agent.roster_counts.values())
    if roster_count >= 40:
        return 0
    return max(agent.budget - (40 - len(agent.roster) - 1), 0)
