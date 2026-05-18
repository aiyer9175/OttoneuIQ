import os
import sys


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
sys.path.insert(0, os.path.abspath(DATA_DIR))

from valuation import (
    ROSTER_SLOTS,
    assign_player_to_roster_slot,
    eligible_roster_slots,
    load_player_pool,
    normalize_player_positions,
)


def empty_roster_counts():
    return {slot: 0 for slot in ROSTER_SLOTS}


def test_shortstop_flows_to_mi_before_bench():
    counts = empty_roster_counts()

    assert assign_player_to_roster_slot(counts, ["SS"]) == "SS"
    assert assign_player_to_roster_slot(counts, ["SS"]) == "MI"
    assert assign_player_to_roster_slot(counts, ["SS"]) == "UTIL"
    assert assign_player_to_roster_slot(counts, ["SS"]) == "Bench"


def test_second_base_uses_mi_when_primary_slot_is_full():
    counts = empty_roster_counts()
    counts["2B"] = ROSTER_SLOTS["2B"]

    assert assign_player_to_roster_slot(counts, ["2B"]) == "MI"


def test_corner_infielder_uses_ci_before_util_or_bench():
    counts = empty_roster_counts()
    counts["1B"] = ROSTER_SLOTS["1B"]

    assert assign_player_to_roster_slot(counts, ["1B"]) == "CI"


def test_outfielder_uses_util_before_bench_when_of_is_full():
    counts = empty_roster_counts()
    counts["OF"] = ROSTER_SLOTS["OF"]

    assert assign_player_to_roster_slot(counts, ["OF"]) == "UTIL"


def test_eligibility_waterfall_removes_duplicate_slots():
    assert eligible_roster_slots(["SS", "2B"]) == ["SS", "2B", "MI", "UTIL"]


def test_invalid_position_defaults_to_util():
    assert normalize_player_positions("MIN") == ["UTIL"]
    assert normalize_player_positions("OF/DH") == ["OF", "DH"]


def test_load_player_pool_merges_duplicate_player_positions(tmp_path):
    batters = tmp_path / "batters.csv"
    pitchers = tmp_path / "pitchers.csv"

    batters.write_text(
        "Name,Team,POS,Dollars\n"
        "Shohei Ohtani,LAD,DH,59\n"
        "Other Hitter,NYY,OF,10\n"
    )
    pitchers.write_text(
        "Name,Team,POS,Dollars\n"
        "Shohei Ohtani,LAD,SP,22\n"
        "Other Pitcher,NYY,RP,8\n"
    )

    pool = load_player_pool(str(batters), str(pitchers))
    ohtani = next(player for player in pool if player["Name"] == "Shohei Ohtani")

    assert ohtani["dollars"] == 59
    assert ohtani["positions"] == ["DH", "SP"]
    assert ohtani["active_eligible"]
    assert not ohtani["is_prospect"]
    assert eligible_roster_slots(ohtani["positions"]) == ["SP", "UTIL"]
