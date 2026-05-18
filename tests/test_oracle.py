import os
import sys


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
sys.path.insert(0, os.path.abspath(DATA_DIR))

from oracle import (
    cap_room,
    cut_penalty,
    is_roster_legal,
    legal_cut_need,
    required_empty_roster_reserve,
)


def test_empty_roster_reserve():
    assert required_empty_roster_reserve(37) == 3
    assert required_empty_roster_reserve(40) == 0
    assert required_empty_roster_reserve(42) == 0


def test_roster_legality_matches_ottoneu_examples():
    assert is_roster_legal(roster_size=37, total_salary=390)
    assert is_roster_legal(roster_size=37, total_salary=397)
    assert not is_roster_legal(roster_size=37, total_salary=398)


def test_cut_penalty_rounds_up_half_salary():
    assert cut_penalty(1) == 1
    assert cut_penalty(7) == 4
    assert cut_penalty(18) == 9


def test_legal_cut_need_respects_cap_penalties():
    assert cap_room(total_salary=390, cap_penalties=0) == 10
    assert legal_cut_need(roster_size=37, total_salary=398) == 1
    assert legal_cut_need(roster_size=37, total_salary=397, cap_penalties=1) == 1
