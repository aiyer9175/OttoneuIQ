import argparse
import math
import os
import sys
import warnings

warnings.filterwarnings("ignore", message="Pandas requires version")

import pandas as pd

from data_sources import resolve_data_paths
from prospect_status import apply_prospect_graduation
from valuation import ROSTER_SLOTS, clean_prospect_name, eligible_roster_slots, normalize_player_positions


SALARY_CAP = 400
TOTAL_ROSTER_LIMIT = 40
ACTIVE_SLOTS = {slot: count for slot, count in ROSTER_SLOTS.items() if slot not in {"Bench", "P"}}
MLB_LEVELS = {"AAA", "MLB"}
DEFAULT_PROSPECT_FILE = os.path.join("data", "Baseball Composite Prospect List 2026 - List.csv")
DEFAULT_MLB_STOCK_FILE = "mlb_stock_values.csv"
ACTIVE_SLOT_FILL_ORDER = ["C", "RP", "MI", "CI", "SS", "2B", "3B", "1B", "OF", "UTIL", "SP"]


def money_to_float(value):
    if pd.isna(value):
        return 0.0
    return float(str(value).replace("$", "").replace(",", "").strip() or 0)


def normalize_name(value):
    return str(value).strip()


def normalized_player_id(value):
    if pd.isna(value):
        return None
    try:
        return str(int(float(value)))
    except ValueError:
        return str(value).strip()


def load_ros_values(hitters_csv, pitchers_csv):
    frames = []
    for path in [hitters_csv, pitchers_csv]:
        df = pd.read_csv(path)
        df["Name"] = df["Name"].apply(normalize_name)
        df["NameKey"] = df["Name"].str.lower()
        df["NameASCIIKey"] = df["NameASCII"].fillna(df["Name"]).apply(normalize_name).str.lower()
        df["PlayerIdKey"] = df["PlayerId"].apply(normalized_player_id)
        df["ROS_Dollars"] = pd.to_numeric(df["Dollars"], errors="coerce").fillna(0)
        df["ROS_Points"] = pd.to_numeric(df["rPTS"], errors="coerce").fillna(0)
        df["ROS_Positions"] = df["POS"].apply(normalize_player_positions)
        frames.append(df[["NameKey", "NameASCIIKey", "PlayerIdKey", "ROS_Dollars", "ROS_Points", "ROS_Positions"]])

    ros = pd.concat(frames, ignore_index=True)
    ros = ros.sort_values("ROS_Dollars", ascending=False)
    return ros.drop_duplicates(subset=["PlayerIdKey", "NameKey"], keep="first")


def load_average_values(avg_csv):
    avg = pd.read_csv(avg_csv)
    avg["Name"] = avg["Name"].apply(normalize_name)
    avg["NameKey"] = avg["Name"].str.lower()
    avg["PlayerIdKey"] = avg["FG MajorLeagueID"].apply(normalized_player_id)
    avg["Avg_Salary"] = avg["Avg Salary"].apply(money_to_float)
    avg["Median_Salary"] = avg["Median Salary"].apply(money_to_float)
    avg["Last10_Salary"] = avg["Last 10"].apply(money_to_float)
    return avg[["NameKey", "PlayerIdKey", "Avg_Salary", "Median_Salary", "Last10_Salary", "Roster%"]]


def load_prospect_levels(prospect_csv):
    prospects = pd.read_csv(prospect_csv)
    prospects["Name"] = prospects["Name"].apply(clean_prospect_name).apply(normalize_name)
    prospects["NameKey"] = prospects["Name"].str.lower()
    prospects["Prospect_Level"] = prospects["Highest Level"].fillna("").astype(str).str.upper()
    prospects["Prospect_Rank"] = pd.to_numeric(prospects["Rank"], errors="coerce")
    return prospects[["NameKey", "Prospect_Level", "Prospect_Rank"]]


def load_or_build_mlb_stock(path=DEFAULT_MLB_STOCK_FILE):
    if path and os.path.exists(path):
        return pd.read_csv(path)
    from mlb_stock import build_mlb_stock

    return build_mlb_stock()


def load_rosters(roster_csv):
    roster = pd.read_csv(roster_csv)
    if {"Team", "Player", "Positions", "Salary"}.issubset(roster.columns):
        return load_simulated_rosters(roster)

    roster["Name"] = roster["Name"].apply(normalize_name)
    roster["NameKey"] = roster["Name"].str.lower()
    roster["PlayerIdKey"] = roster["FG MajorLeagueID"].apply(normalized_player_id)
    roster["Salary"] = roster["Salary"].apply(money_to_float)
    roster["Roster_Positions"] = roster["Position(s)"].apply(normalize_player_positions)
    return roster


def load_simulated_rosters(roster):
    simulated = roster.copy()
    simulated["Team Name"] = simulated["Team"].astype(str)
    simulated["Name"] = simulated["Player"].apply(normalize_name)
    simulated["NameKey"] = simulated["Name"].str.lower()
    simulated["PlayerIdKey"] = None
    simulated["MLB Team"] = ""
    simulated["Salary"] = pd.to_numeric(simulated["Salary"], errors="coerce").fillna(1)
    simulated["Roster_Positions"] = simulated["Positions"].apply(normalize_player_positions)
    simulated["FG MajorLeagueID"] = None
    simulated["FG MinorLeagueID"] = None
    simulated["ottoneu ID"] = None
    return simulated


def merge_player_context(roster, ros, avg, prospects):
    ros_by_id = ros[ros["PlayerIdKey"].notna()].drop_duplicates("PlayerIdKey")
    merged = roster.merge(
        ros_by_id[["PlayerIdKey", "ROS_Dollars", "ROS_Points", "ROS_Positions"]],
        on="PlayerIdKey",
        how="left",
    )
    fill_missing_ros(merged, ros, "NameKey")
    ros_ascii = ros[["NameASCIIKey", "ROS_Dollars", "ROS_Points", "ROS_Positions"]].rename(
        columns={"NameASCIIKey": "NameKey"}
    )
    fill_missing_ros(merged, ros_ascii, "NameKey")

    avg_by_id = avg[avg["PlayerIdKey"].notna()].drop_duplicates("PlayerIdKey")
    merged = merged.merge(
        avg_by_id[["PlayerIdKey", "Avg_Salary", "Median_Salary", "Last10_Salary", "Roster%"]],
        on="PlayerIdKey",
        how="left",
    )
    fill_missing_avg(merged, avg)

    merged = merged.merge(prospects, on="NameKey", how="left")
    merged["ROS_Dollars"] = pd.to_numeric(merged["ROS_Dollars"], errors="coerce").fillna(0)
    merged["ROS_Points"] = pd.to_numeric(merged["ROS_Points"], errors="coerce").fillna(0)
    merged["Avg_Salary"] = pd.to_numeric(merged["Avg_Salary"], errors="coerce").fillna(0)
    merged["Median_Salary"] = pd.to_numeric(merged["Median_Salary"], errors="coerce").fillna(0)
    merged["Last10_Salary"] = pd.to_numeric(merged["Last10_Salary"], errors="coerce").fillna(0)
    merged["Roster%"] = pd.to_numeric(merged["Roster%"], errors="coerce").fillna(0)
    merged["Is_Prospect"] = merged["Prospect_Rank"].notna()
    merged["MLB_Level"] = (~merged["Is_Prospect"]) | merged["Prospect_Level"].isin(MLB_LEVELS)
    merged["Effective_Positions"] = merged.apply(resolve_positions, axis=1)
    merged["Future_Value"] = merged.apply(future_value, axis=1)
    merged["ROS_Surplus"] = merged["ROS_Dollars"] - merged["Salary"]
    merged["Market_Surplus"] = merged["Avg_Salary"] - merged["Salary"]
    merged["Future_Surplus"] = merged["Future_Value"] - merged["Salary"]
    merged["Cut_Penalty"] = merged["Salary"].apply(lambda salary: int(math.ceil(salary / 2.0)))
    merged["Cap_Recovered"] = merged["Salary"] - merged["Cut_Penalty"]
    return merged


def apply_mlb_stock_context(merged, stock_path=DEFAULT_MLB_STOCK_FILE):
    stock = load_or_build_mlb_stock(stock_path)
    if stock.empty or "PlayerIdKey" not in merged.columns:
        return merged

    stock_cols = [
        "PlayerIdKey", "Preseason_Value", "ROS_Value", "Projection_Change",
        "MLB_Stock_Change", "YTD_Value", "YTD_ROS_Gap", "Banked_Value_Signal",
        "Skill_Score", "Role_Change", "Stock_Label", "Confidence_Label",
        "Player_Type", "YTD_PA", "YTD_IP",
    ]
    for col in stock_cols:
        if col not in stock.columns:
            stock[col] = pd.NA
    stock = stock[stock["PlayerIdKey"].notna()].drop_duplicates("PlayerIdKey")
    merged = merged.merge(stock[stock_cols], on="PlayerIdKey", how="left", suffixes=("", "_MLB"))

    merged["Model_Future_Value"] = merged["Future_Value"]
    mlb_value = pd.to_numeric(merged["ROS_Value"], errors="coerce").notna()
    merged.loc[mlb_value, "Future_Value"] = pd.to_numeric(merged.loc[mlb_value, "ROS_Value"], errors="coerce")
    merged["Future_Surplus"] = merged["Future_Value"] - merged["Salary"]
    merged["Stock_Change"] = pd.to_numeric(merged["MLB_Stock_Change"], errors="coerce").fillna(0)
    merged["YTD_Value"] = pd.to_numeric(merged["YTD_Value"], errors="coerce").fillna(0)
    merged["YTD_ROS_Gap"] = pd.to_numeric(merged["YTD_ROS_Gap"], errors="coerce").fillna(0)
    merged["Banked_Value_Signal"] = pd.to_numeric(merged["Banked_Value_Signal"], errors="coerce").fillna(0)
    merged["Skill_Score"] = pd.to_numeric(merged["Skill_Score"], errors="coerce").fillna(0.5)
    merged["Role_Change"] = merged["Role_Change"].fillna("Standard")
    merged["Stock_Label"] = merged["Stock_Label"].fillna("Standard")
    merged["Stock_Confidence"] = merged["Confidence_Label"].fillna("Standard")
    merged = apply_prospect_graduation(merged)
    return merged


def fill_missing_ros(merged, ros, key_col):
    missing = merged["ROS_Dollars"].isna()
    if not missing.any():
        return
    ros_by_name = ros.drop_duplicates(key_col)
    fallback = merged.loc[missing, [key_col]].merge(ros_by_name, on=key_col, how="left")
    target_index = merged.index[missing]
    for row_num, idx in enumerate(target_index):
        if pd.isna(merged.at[idx, "ROS_Dollars"]) and pd.notna(fallback.at[row_num, "ROS_Dollars"]):
            for col in ["ROS_Dollars", "ROS_Points", "ROS_Positions"]:
                merged.at[idx, col] = fallback.at[row_num, col]


def fill_missing_avg(merged, avg):
    missing = merged["Avg_Salary"].isna()
    if not missing.any():
        return
    avg_by_name = avg.drop_duplicates("NameKey")
    fallback = merged.loc[missing, ["NameKey"]].merge(avg_by_name, on="NameKey", how="left")
    target_index = merged.index[missing]
    for row_num, idx in enumerate(target_index):
        if pd.isna(merged.at[idx, "Avg_Salary"]) and pd.notna(fallback.at[row_num, "Avg_Salary"]):
            for col in ["Avg_Salary", "Median_Salary", "Last10_Salary", "Roster%"]:
                merged.at[idx, col] = fallback.at[row_num, col]


def resolve_positions(row):
    ros_positions = row.get("ROS_Positions")
    if isinstance(ros_positions, list) and ros_positions:
        return ros_positions
    return row["Roster_Positions"]


def future_value(row):
    ros_value = float(row["ROS_Dollars"])
    market_value = float(row["Avg_Salary"])
    if row["Is_Prospect"]:
        return max(ros_value, market_value * 0.85)
    return (ros_value * 0.75) + (market_value * 0.25)


def assign_active_slots(team_df):
    team_df = team_df.copy()
    team_df["Active_Slot"] = "Reserve"
    used = set()
    candidates = team_df[team_df["MLB_Level"]].copy()

    for slot in ACTIVE_SLOT_FILL_ORDER:
        if slot not in ACTIVE_SLOTS:
            continue
        for _ in range(ACTIVE_SLOTS[slot]):
            eligible = []
            for idx, row in candidates.iterrows():
                if idx in used:
                    continue
                if slot in eligible_roster_slots(row["Effective_Positions"]):
                    eligible.append((idx, float(row["Future_Value"])))
            if not eligible:
                break
            idx, _ = max(eligible, key=lambda item: item[1])
            used.add(idx)
            team_df.loc[idx, "Active_Slot"] = slot

    return team_df


def recommend_keep_cut(row):
    if row.get("Stock_Label") == "YTD Breakout, Projection Skeptical":
        if row["Future_Surplus"] <= -10:
            return "SHOP"
        return "HOLD"
    if row.get("YTD_Value", 0) >= max(row["Salary"] * 0.8, 15) and row.get("Skill_Score", 0.5) >= 0.58:
        if row["Future_Surplus"] <= -10:
            return "SHOP"
        return "HOLD"
    if row["Active_Slot"] != "Reserve" and row["Market_Surplus"] >= -5:
        return "KEEP"
    if row["Active_Slot"] != "Reserve" and row["Future_Surplus"] >= -10:
        return "KEEP"
    if row["Is_Prospect"] and row["Prospect_Level"] not in MLB_LEVELS and row["Avg_Salary"] >= max(row["Salary"], 3):
        return "KEEP"
    if row["Future_Surplus"] >= 5:
        return "KEEP"
    if row["Active_Slot"] != "Reserve" and row["Market_Surplus"] >= -12:
        return "HOLD"
    if row["Future_Surplus"] <= -10 and row["Market_Surplus"] <= -8 and row["Cap_Recovered"] >= 4:
        return "CUT"
    if row["Future_Surplus"] <= -5 and row["Market_Surplus"] <= -5 and row["Cap_Recovered"] >= 3:
        return "SHOP"
    if row["ROS_Dollars"] <= 0 and row["Avg_Salary"] <= 2 and row["Salary"] >= 3:
        return "CUT"
    return "HOLD"


def build_keep_cut_report(merged):
    report = pd.concat(
        [assign_active_slots(team_df) for _, team_df in merged.groupby("Team Name")],
        ignore_index=True,
    )
    report["Recommendation"] = report.apply(recommend_keep_cut, axis=1)
    report["Positions"] = report["Effective_Positions"].apply(lambda positions: "/".join(positions))
    columns = [
        "Team Name", "Name", "PlayerIdKey", "MLB Team", "Positions", "Salary", "ROS_Dollars", "Avg_Salary",
        "Future_Value", "Future_Surplus", "ROS_Surplus", "Market_Surplus", "Stock_Change",
        "YTD_Value", "YTD_ROS_Gap", "Banked_Value_Signal", "Stock_Label", "Role_Change",
        "Active_Slot",
        "MLB_Level", "Is_Prospect", "Prospect_Level", "Prospect_Rank", "Cut_Penalty",
        "Cap_Recovered", "Recommendation",
    ]
    for col in columns:
        if col not in report.columns:
            report[col] = pd.NA
    return report[columns].sort_values(["Team Name", "Recommendation", "Future_Surplus"], ascending=[True, True, False])


def build_team_summary(report):
    summary = report.groupby("Team Name").agg(
        Roster_Size=("Name", "count"),
        Salary_Used=("Salary", "sum"),
        MLB_Level_Count=("MLB_Level", "sum"),
        Active_Filled=("Active_Slot", lambda slots: (slots != "Reserve").sum()),
        Prospect_Count=("Is_Prospect", "sum"),
        Cut_Count=("Recommendation", lambda recs: (recs == "CUT").sum()),
        Shop_Count=("Recommendation", lambda recs: (recs == "SHOP").sum()),
    ).reset_index()
    summary["Cap_Room"] = SALARY_CAP - summary["Salary_Used"]
    summary["Required_Reserve"] = (TOTAL_ROSTER_LIMIT - summary["Roster_Size"]).clip(lower=0)
    summary["Cap_Legal"] = summary["Cap_Room"] >= summary["Required_Reserve"]
    summary["Active_Lineup_Complete"] = summary["Active_Filled"] >= sum(ACTIVE_SLOTS.values())
    return summary.sort_values("Team Name")


def build_arbitration_report(report, target_team=None, limit=75):
    candidates = report.copy()
    if target_team:
        candidates = candidates[candidates["Team Name"] != target_team]
    stock_change = pd.to_numeric(candidates.get("Stock_Change", pd.Series(0, index=candidates.index)), errors="coerce").fillna(0)
    banked_signal = pd.to_numeric(
        candidates.get("Banked_Value_Signal", pd.Series(0, index=candidates.index)),
        errors="coerce",
    ).fillna(0)
    candidates["Arb_Priority"] = (
        candidates["Future_Surplus"]
        + candidates["Market_Surplus"].clip(lower=-5) * 0.35
        + stock_change.clip(lower=0, upper=15) * 0.4
        + banked_signal.clip(lower=0, upper=5) * 0.7
        + candidates["Active_Slot"].ne("Reserve").astype(int) * 3
    )
    candidates = candidates[candidates["Arb_Priority"] > 0]
    columns = [
        "Team Name", "Name", "Positions", "Salary", "Future_Value", "Future_Surplus",
        "Avg_Salary", "Market_Surplus", "Stock_Change", "YTD_Value", "Stock_Label",
        "Active_Slot", "Is_Prospect", "Arb_Priority",
    ]
    for col in columns:
        if col not in candidates.columns:
            candidates[col] = pd.NA
    return candidates[columns].sort_values("Arb_Priority", ascending=False).head(limit)


def build_reports(rosters, hitters, pitchers, avg, prospects, mlb_stock=DEFAULT_MLB_STOCK_FILE):
    roster = load_rosters(rosters)
    ros = load_ros_values(hitters, pitchers)
    avg_values = load_average_values(avg)
    prospect_levels = load_prospect_levels(prospects)
    merged = merge_player_context(roster, ros, avg_values, prospect_levels)
    merged = apply_mlb_stock_context(merged, mlb_stock)
    report = build_keep_cut_report(merged)
    summary = build_team_summary(report)
    return report, summary


def print_table(df, columns, sort_by=None, ascending=False, limit=None):
    output = df.copy()
    if sort_by:
        output = output.sort_values(sort_by, ascending=ascending)
    if limit:
        output = output.head(limit)
    if output.empty:
        print("No rows found.")
        return
    print(output[columns].to_string(index=False, float_format=lambda value: f"{value:.2f}"))


def print_team_keepcut(report, team, limit=None):
    team_report = report[report["Team Name"].eq(team)]
    columns = [
        "Name", "Salary", "Future_Value", "Future_Surplus", "YTD_Value",
        "Stock_Label", "Active_Slot", "Recommendation",
    ]
    print_table(team_report, columns, sort_by=["Recommendation", "Future_Surplus"], ascending=[True, False], limit=limit)


def print_team_arbtargets(report, team, limit=25):
    arb = build_arbitration_report(report, target_team=team, limit=limit)
    columns = [
        "Team Name", "Name", "Positions", "Salary", "Future_Value",
        "Future_Surplus", "YTD_Value", "Stock_Label", "Avg_Salary",
        "Market_Surplus", "Arb_Priority",
    ]
    print_table(arb, columns)


def print_underpriced(report, limit=25):
    columns = [
        "Team Name", "Name", "Positions", "Salary", "Future_Value",
        "Future_Surplus", "YTD_Value", "Stock_Label", "Avg_Salary",
        "Market_Surplus", "Active_Slot",
    ]
    print_table(report, columns, sort_by="Future_Surplus", ascending=False, limit=limit)


def resolve_team_name(report, team):
    teams = sorted(report["Team Name"].dropna().unique())
    if team in teams:
        return team

    normalized = team.strip().lower()
    matches = [candidate for candidate in teams if candidate.strip().lower() == normalized]
    if len(matches) == 1:
        return matches[0]

    matches = [candidate for candidate in teams if normalized in candidate.strip().lower()]
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise ValueError(f"Ambiguous team '{team}'. Matches: {matches}")
    raise ValueError(f"Unknown team '{team}'. Available teams: {teams}")


def add_data_args(parser):
    parser.add_argument("--rosters", default="current_rosters.csv")
    parser.add_argument("--hitters", default="hitters_ros.csv")
    parser.add_argument("--pitchers", default="pitchers_ros.csv")
    parser.add_argument("--avg", default="fgpts_avgvalues.csv")
    parser.add_argument("--prospects", default=DEFAULT_PROSPECT_FILE)
    parser.add_argument("--data-source", choices=["static", "cache"], default=None)


def parse_args():
    parser = argparse.ArgumentParser(description="Post-auction keep/cut and arbitration analyzer.")
    add_data_args(parser)
    parser.add_argument("--team", default=None, help="Optional team name for a focused keep/cut file and arbitration exclusions.")
    parser.add_argument("--out-dir", default="post_auction_reports")
    return parser.parse_args()


def main():
    if len(sys.argv) > 1 and sys.argv[1] in {"team", "underpriced", "summary", "teams"}:
        command_main()
        return

    args = parse_args()
    if args.data_source:
        paths = resolve_data_paths(args.data_source)
        args.rosters = paths.rosters
        args.hitters = paths.hitters_ros
        args.pitchers = paths.pitchers_ros
        args.avg = paths.avg_values
        args.prospects = paths.prospects
    os.makedirs(args.out_dir, exist_ok=True)

    report, summary = build_reports(args.rosters, args.hitters, args.pitchers, args.avg, args.prospects)
    arbitration = build_arbitration_report(report, target_team=args.team)

    report.to_csv(os.path.join(args.out_dir, "keep_cut_report.csv"), index=False, float_format="%.2f")
    summary.to_csv(os.path.join(args.out_dir, "team_summary.csv"), index=False, float_format="%.2f")
    arbitration.to_csv(os.path.join(args.out_dir, "arbitration_targets.csv"), index=False, float_format="%.2f")

    if args.team:
        team_report = report[report["Team Name"] == args.team]
        team_path = os.path.join(args.out_dir, f"{args.team.replace(' ', '_').lower()}_keep_cut.csv")
        team_report.to_csv(team_path, index=False, float_format="%.2f")

    print(f"Wrote reports to {args.out_dir}")
    print(f"Teams analyzed: {summary.shape[0]}")
    print(f"Roster rows analyzed: {report.shape[0]}")


def command_main():
    parser = argparse.ArgumentParser(description="Print post-auction views in the terminal.")
    add_data_args(parser)
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("command", choices=["team", "underpriced", "summary", "teams"])
    parser.add_argument("args", nargs="*")
    args = parser.parse_args()

    report, summary = build_reports(args.rosters, args.hitters, args.pitchers, args.avg, args.prospects)

    if args.command == "teams":
        print("\n".join(summary["Team Name"].astype(str).tolist()))
        return

    if args.command == "summary":
        columns = [
            "Team Name", "Roster_Size", "Salary_Used", "Cap_Room",
            "Active_Filled", "Active_Lineup_Complete", "Cut_Count", "Shop_Count",
        ]
        print_table(summary, columns)
        return

    if args.command == "underpriced":
        limit = int(args.args[0]) if args.args and args.args[0].isdigit() else args.limit
        print_underpriced(report, limit=limit)
        return

    if args.command == "team":
        if len(args.args) < 2:
            raise SystemExit('Usage: post_auction.py team "Team Name" keepcut|arbtargets|summary [limit]')
        team = resolve_team_name(report, args.args[0])
        view = args.args[1].lower()
        limit = int(args.args[2]) if len(args.args) >= 3 and args.args[2].isdigit() else args.limit

        if view in {"keepcut", "cuts", "verdicts"}:
            print_team_keepcut(report, team, limit=limit)
            return
        if view in {"arbtargets", "arbitration", "arb"}:
            print_team_arbtargets(report, team, limit=limit)
            return
        if view == "summary":
            columns = [
                "Team Name", "Roster_Size", "Salary_Used", "Cap_Room",
                "Active_Filled", "Active_Lineup_Complete", "Cut_Count", "Shop_Count",
            ]
            print_table(summary[summary["Team Name"].eq(team)], columns)
            return
        raise SystemExit(f"Unknown team view '{view}'. Use keepcut, arbtargets, or summary.")


if __name__ == "__main__":
    main()
