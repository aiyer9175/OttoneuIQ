import argparse
import contextlib
import io
import os
import random
import warnings

import numpy as np

warnings.filterwarnings("ignore", message="Pandas requires version")

import pandas as pd

from live_auction import LiveAuctionRoom, active_lineup_open_count, roster_size
from valuation import load_player_pool
from bandit_policy import LinUCBBidPolicy
from data_sources import resolve_data_paths


DEFAULT_PROSPECT_FILE = "Baseball Composite Prospect List 2026 - List.csv"


def classify_strategy(row):
    if row["Prospects"] >= 10 or row["Prospect_Salary"] >= 25:
        return "prospect-heavy"
    if row["Stars_40"] >= 4 and row["Cheap_5"] >= 24:
        return "extreme studs/duds"
    if row["Stars_40"] >= 3 and row["Cheap_5"] >= 20:
        return "studs/duds"
    if row["Stars_40"] <= 1 and row["Mid_6_25"] >= 18:
        return "balanced depth"
    if row["Stars_40"] >= 2 and row["Mid_6_25"] >= 12:
        return "stars and depth"
    return "mixed"


def summarize_team(sim_id, team_idx, room):
    team = room.team_label(team_idx)
    rows = [row for row in room.roster_rows if row["Team"] == team]
    salaries = [float(row["Salary"]) for row in rows]
    values = [float(row["Value"]) for row in rows]
    prospect_rows = [row for row in rows if row["Is_Prospect"]]
    active_rows = [row for row in rows if row["Slot"] != "Bench"]

    summary = {
        "Sim": sim_id,
        "Team": team,
        "Players": len(rows),
        "Budget_Left": room.agents[team_idx].budget,
        "Open_Active": active_lineup_open_count(room.agents[team_idx]),
        "Stars_40": sum(salary >= 40 for salary in salaries),
        "Cheap_5": sum(salary <= 5 for salary in salaries),
        "Mid_6_25": sum(6 <= salary <= 25 for salary in salaries),
        "Prospects": len(prospect_rows),
        "Prospect_Salary": sum(float(row["Salary"]) for row in prospect_rows),
        "Active_Salary": sum(float(row["Salary"]) for row in active_rows),
        "Bench_Salary": sum(float(row["Salary"]) for row in rows if row["Slot"] == "Bench"),
        "Total_Value": round(sum(values), 2),
        "Total_Salary": sum(salaries),
        "Total_Surplus": round(sum(float(row["Surplus"]) for row in rows), 2),
        "Alpha": room.agents[team_idx].alpha,
        "Beta": room.agents[team_idx].beta,
    }
    summary["Strategy"] = classify_strategy(summary)
    return summary


def run_single_sim(pool, sim_id, seed=None, bid_policy=None, max_picks=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    room = LiveAuctionRoom(pool=pool, human_team=0, auto_human=True, bid_policy=bid_policy, max_picks=max_picks)
    with contextlib.redirect_stdout(io.StringIO()):
        if max_picks:
            room.simulate_picks(max_picks)
        else:
            room.simulate_to_end()
    return [summarize_team(sim_id, idx, room) for idx in range(12)], room


def run_auction_sims(
    sim_count=20,
    batters="batters_auctioncalc.csv",
    pitchers="pitchers_auctioncalc.csv",
    prospects=DEFAULT_PROSPECT_FILE,
    seed=42,
    max_picks=None,
    progress_callback=None,
):
    pool = load_player_pool(batters, pitchers, prospects)
    rows = []
    for sim_idx in range(1, sim_count + 1):
        sim_seed = None if seed is None else seed + sim_idx
        sim_rows, _ = run_single_sim(pool, sim_idx, seed=sim_seed, max_picks=max_picks)
        rows.extend(sim_rows)
        if progress_callback:
            progress_callback(sim_idx, sim_count)
    team_runs = pd.DataFrame(rows)
    strategy_summary = (
        team_runs["Strategy"]
        .value_counts()
        .rename_axis("Strategy")
        .reset_index(name="Team_Sims")
    )
    strategy_summary["Share"] = strategy_summary["Team_Sims"] / len(team_runs)
    metrics = [
        "Stars_40", "Cheap_5", "Mid_6_25", "Prospects", "Prospect_Salary",
        "Active_Salary", "Bench_Salary", "Total_Value", "Total_Surplus",
        "Alpha", "Beta",
    ]
    metric_summary = (
        team_runs.groupby("Strategy")[metrics]
        .mean()
        .round(2)
        .reset_index()
        .sort_values("Total_Surplus", ascending=False)
    )
    return team_runs, strategy_summary, metric_summary


def run_bandit_training(
    sim_count=20,
    batters="batters_auctioncalc.csv",
    pitchers="pitchers_auctioncalc.csv",
    prospects=DEFAULT_PROSPECT_FILE,
    seed=42,
    alpha=0.8,
    max_picks=None,
    progress_callback=None,
):
    pool = load_player_pool(batters, pitchers, prospects)
    bid_policy = LinUCBBidPolicy(alpha=alpha, seed=seed)
    rows = []
    decision_rows = []
    for sim_idx in range(1, sim_count + 1):
        sim_seed = None if seed is None else seed + sim_idx
        sim_rows, room = run_single_sim(pool, sim_idx, seed=sim_seed, bid_policy=bid_policy, max_picks=max_picks)
        rows.extend(sim_rows)
        decision_rows.extend(room.bandit_decision_rows)
        if progress_callback:
            progress_callback(sim_idx, sim_count)

    team_runs = pd.DataFrame(rows)
    decisions = pd.DataFrame(decision_rows)
    strategy_summary = (
        team_runs["Strategy"]
        .value_counts()
        .rename_axis("Strategy")
        .reset_index(name="Team_Sims")
    )
    strategy_summary["Share"] = strategy_summary["Team_Sims"] / len(team_runs)
    action_summary = bid_policy.action_summary()
    return team_runs, strategy_summary, action_summary, decisions


def main():
    parser = argparse.ArgumentParser(description="Run many Ottoneu auction sims and summarize roster strategy shapes.")
    parser.add_argument("--sims", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batters", default="batters_auctioncalc.csv")
    parser.add_argument("--pitchers", default="pitchers_auctioncalc.csv")
    parser.add_argument("--prospects", default=DEFAULT_PROSPECT_FILE)
    parser.add_argument("--data-source", choices=["static", "cache"], default=None)
    parser.add_argument("--out", default="auction_sim_strategy_runs.csv")
    parser.add_argument("--max-picks", type=int, default=None, help="Optional partial-auction cap for faster experiments.")
    parser.add_argument("--bandit", action="store_true", help="Train a LinUCB bid-aggressiveness policy across simulations.")
    parser.add_argument("--decisions-out", default="auction_bandit_decisions.csv")
    args = parser.parse_args()

    if not os.path.exists(args.batters) and os.path.exists(os.path.join("data", args.batters)):
        os.chdir("data")
    if args.data_source:
        paths = resolve_data_paths(args.data_source)
        args.batters = paths.batters_auction
        args.pitchers = paths.pitchers_auction
        args.prospects = paths.prospects

    if args.bandit:
        team_runs, strategy_summary, action_summary, decisions = run_bandit_training(
            sim_count=args.sims,
            batters=args.batters,
            pitchers=args.pitchers,
            prospects=args.prospects,
            seed=args.seed,
            max_picks=args.max_picks,
        )
        team_runs.to_csv(args.out, index=False)
        decisions.to_csv(args.decisions_out, index=False)
        print(f"Wrote {args.out} with {len(team_runs)} team-sim rows.")
        print(f"Wrote {args.decisions_out} with {len(decisions)} bandit purchase decisions.")
        print("\nStrategy frequency")
        print(strategy_summary.to_string(index=False, float_format=lambda value: f"{value:.2%}"))
        print("\nBandit action summary")
        print(action_summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
        return

    team_runs, strategy_summary, metric_summary = run_auction_sims(
        sim_count=args.sims,
        batters=args.batters,
        pitchers=args.pitchers,
        prospects=args.prospects,
        seed=args.seed,
        max_picks=args.max_picks,
    )
    team_runs.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with {len(team_runs)} team-sim rows.")
    print("\nStrategy frequency")
    print(strategy_summary.to_string(index=False, float_format=lambda value: f"{value:.2%}"))
    print("\nAverage metrics by strategy")
    print(metric_summary.to_string(index=False))


if __name__ == "__main__":
    main()
