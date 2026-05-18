import argparse
import math
import os

import pandas as pd

from valuation import TOTAL_ROSTER_LIMIT


SALARY_CAP = 400
DEFAULT_ROSTER_FILE = "final_team_rosters.csv"
DEFAULT_DRAFT_FILE = "full_480_draft_results.csv"


def required_empty_roster_reserve(roster_size):
    return max(TOTAL_ROSTER_LIMIT - roster_size, 0)


def cut_penalty(salary):
    return int(math.ceil(float(salary) / 2.0))


def cap_room(total_salary, cap_penalties=0):
    return SALARY_CAP - int(total_salary) - int(cap_penalties)


def is_roster_legal(roster_size, total_salary, cap_penalties=0):
    return cap_room(total_salary, cap_penalties) >= required_empty_roster_reserve(roster_size)


def legal_cut_need(roster_size, total_salary, cap_penalties=0):
    reserve = required_empty_roster_reserve(roster_size)
    return max(reserve - cap_room(total_salary, cap_penalties), 0)


def load_roster_source(roster_csv=DEFAULT_ROSTER_FILE, draft_csv=DEFAULT_DRAFT_FILE):
    if os.path.exists(roster_csv):
        return pd.read_csv(roster_csv)

    if not os.path.exists(draft_csv):
        raise FileNotFoundError(
            f"Could not find {roster_csv} or fallback draft file {draft_csv}."
        )

    draft_df = pd.read_csv(draft_csv)
    return pd.DataFrame({
        "Team": draft_df["Winner"],
        "Player": draft_df["Player"],
        "Slot": "Unknown",
        "Positions": "",
        "Salary": draft_df["Price"],
        "Value": draft_df["Value"],
        "Surplus": draft_df["Value"] - draft_df["Price"],
        "Is_Prospect": False,
    })


def normalize_roster(df):
    normalized = df.copy()
    rename_map = {
        "Winner": "Team",
        "Price": "Salary",
        "Dollars": "Value",
    }
    normalized = normalized.rename(columns=rename_map)

    required_columns = {"Team", "Player", "Salary", "Value"}
    missing = required_columns - set(normalized.columns)
    if missing:
        raise ValueError(f"Roster data is missing required columns: {sorted(missing)}")

    if "Slot" not in normalized.columns:
        normalized["Slot"] = "Unknown"
    if "Positions" not in normalized.columns:
        normalized["Positions"] = ""
    if "Is_Prospect" not in normalized.columns:
        normalized["Is_Prospect"] = False

    normalized["Salary"] = pd.to_numeric(normalized["Salary"], errors="coerce").fillna(1)
    normalized["Value"] = pd.to_numeric(normalized["Value"], errors="coerce").fillna(0)
    normalized["Surplus"] = (normalized["Value"] - normalized["Salary"]).round(2)
    normalized["Is_Prospect"] = normalized["Is_Prospect"].astype(bool)
    return normalized


def score_cut_candidate(row, team_is_illegal=False):
    salary = float(row["Salary"])
    value = float(row["Value"])
    surplus = float(row["Surplus"])
    penalty = cut_penalty(salary)
    cap_recovered = salary - penalty

    cut_score = cap_recovered - max(value, 0)
    if team_is_illegal:
        cut_score += min(cap_recovered, 10)
    if row.get("Slot") != "Bench":
        cut_score -= 3
    if bool(row.get("Is_Prospect")) and value >= 3:
        cut_score -= 4

    if surplus >= 5:
        recommendation = "KEEP"
    elif cut_score >= 4:
        recommendation = "CUT"
    elif surplus < 0 and cap_recovered >= 4:
        recommendation = "SHOP"
    else:
        recommendation = "HOLD"

    return pd.Series({
        "Cut_Penalty": penalty,
        "Cap_Recovered": int(cap_recovered),
        "Cut_Score": round(cut_score, 2),
        "Recommendation": recommendation,
    })


def advise_team(roster_df, team, cap_penalties=0):
    normalized = normalize_roster(roster_df)
    team_df = normalized[normalized["Team"].astype(str) == str(team)].copy()
    if team_df.empty:
        available = sorted(normalized["Team"].astype(str).unique())
        raise ValueError(f"No rows found for {team}. Available teams: {available}")

    roster_size = len(team_df)
    total_salary = int(team_df["Salary"].sum())
    illegal = not is_roster_legal(roster_size, total_salary, cap_penalties)

    advice = team_df.join(
        team_df.apply(score_cut_candidate, axis=1, team_is_illegal=illegal)
    )
    recommendation_rank = {"CUT": 0, "SHOP": 1, "HOLD": 2, "KEEP": 3}
    advice["_Recommendation_Rank"] = advice["Recommendation"].map(recommendation_rank)
    advice = advice.sort_values(
        by=["_Recommendation_Rank", "Cut_Score", "Surplus"],
        ascending=[True, False, True],
    ).drop(columns=["_Recommendation_Rank"]).reset_index(drop=True)

    advice.attrs["roster_size"] = roster_size
    advice.attrs["total_salary"] = total_salary
    advice.attrs["cap_room"] = cap_room(total_salary, cap_penalties)
    advice.attrs["required_reserve"] = required_empty_roster_reserve(roster_size)
    advice.attrs["legal_cut_need"] = legal_cut_need(roster_size, total_salary, cap_penalties)
    return advice


def main():
    parser = argparse.ArgumentParser(description="Ottoneu in-season cut/keep advisor.")
    parser.add_argument("--team", default="Team 1", help="Team label, e.g. 'Team 1'.")
    parser.add_argument("--rosters", default=DEFAULT_ROSTER_FILE, help="Roster CSV path.")
    parser.add_argument("--draft", default=DEFAULT_DRAFT_FILE, help="Fallback draft CSV path.")
    parser.add_argument("--cap-penalties", type=int, default=0, help="Existing cap penalties.")
    parser.add_argument("--output", default=None, help="Optional output CSV path.")
    args = parser.parse_args()

    roster_df = load_roster_source(args.rosters, args.draft)
    advice = advise_team(roster_df, args.team, args.cap_penalties)

    output_path = args.output or f"{args.team.lower().replace(' ', '_')}_cut_keep_advice.csv"
    advice.to_csv(output_path, index=False, float_format="%.2f")

    print(f"--- Oracle Cut/Keep Advisor: {args.team} ---")
    print(f"Roster Size: {advice.attrs['roster_size']} / {TOTAL_ROSTER_LIMIT}")
    print(f"Salary Used: ${advice.attrs['total_salary']}")
    print(f"Cap Room: ${advice.attrs['cap_room']}")
    print(f"Required Empty-Spot Reserve: ${advice.attrs['required_reserve']}")
    print(f"Legal Cut Need: ${advice.attrs['legal_cut_need']}")
    print(f"Advice written to {output_path}")


if __name__ == "__main__":
    main()
