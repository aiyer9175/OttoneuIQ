import argparse
import os
import warnings

warnings.filterwarnings("ignore", message="Pandas requires version")
import pandas as pd

from post_auction import ACTIVE_SLOTS, resolve_team_name
from data_sources import resolve_data_paths
from value_engine import build_player_value_table


SLOT_ORDER = {
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
    "Reserve": 99,
}
CATEGORY_COLUMNS = {
    "R": ["R", "Runs"],
    "HR": ["HR"],
    "RBI": ["RBI"],
    "SB": ["SB"],
    "AVG": ["AVG", "AVG+"],
    "OBP": ["OBP"],
    "K": ["K", "SO"],
    "SV": ["SV", "Saves"],
    "ERA": ["ERA"],
    "WHIP": ["WHIP"],
}
AUDIT_VALUE_FLOOR = -5.0
SUPERSTAR_VALUE = 40.0


def audit_value(value, floor=AUDIT_VALUE_FLOOR):
    return max(float(pd.to_numeric(value, errors="coerce") if not pd.isna(value) else 0), floor)


def add_audit_values(values):
    enriched = values.copy()
    enriched["Current_Value"] = pd.to_numeric(enriched["Current_Value"], errors="coerce").fillna(0)
    enriched["Audit_Value"] = enriched["Current_Value"].apply(audit_value)
    enriched["Audit_Surplus"] = enriched["Audit_Value"] - pd.to_numeric(enriched["Salary"], errors="coerce").fillna(0)
    return enriched


def slot_sort_key(slot):
    return SLOT_ORDER.get(str(slot), 99)


def active_slot_labels():
    labels = []
    for slot, count in sorted(ACTIVE_SLOTS.items(), key=lambda item: slot_sort_key(item[0])):
        for idx in range(count):
            labels.append(slot if count == 1 else f"{slot}{idx + 1}")
    return labels


def active_player_rows(team_df):
    rows = []
    for slot, count in sorted(ACTIVE_SLOTS.items(), key=lambda item: slot_sort_key(item[0])):
        slot_rows = (
            team_df[team_df["Active_Slot"].eq(slot)]
            .sort_values("Current_Value", ascending=False)
            .to_dict("records")
        )
        for idx in range(count):
            row = slot_rows[idx] if idx < len(slot_rows) else {}
            rows.append({
                "Slot": slot if count == 1 else f"{slot}{idx + 1}",
                "Position": slot,
                "Name": row.get("Name", ""),
                "Salary": row.get("Salary", ""),
                "Current_Value": row.get("Current_Value", 0.0),
                "Audit_Value": row.get("Audit_Value", audit_value(row.get("Current_Value", 0.0))),
                "Current_Surplus": row.get("Current_Surplus", 0.0),
                "Audit_Surplus": row.get("Audit_Surplus", row.get("Current_Surplus", 0.0)),
                "Stock_Label": row.get("Stock_Label", ""),
                "Positions": row.get("Positions", ""),
            })
    return pd.DataFrame(rows)


def positional_baselines(values):
    active = values[values["Active_Slot"].ne("Reserve")].copy()
    if active.empty:
        return pd.DataFrame(columns=["Position", "League_Avg_Value", "League_Median_Value", "Starter_Count"])
    active = add_audit_values(active)
    return (
        active.groupby("Active_Slot")
        .agg(
            League_Avg_Value=("Audit_Value", "mean"),
            League_Median_Value=("Audit_Value", "median"),
            Starter_Count=("Name", "count"),
        )
        .reset_index()
        .rename(columns={"Active_Slot": "Position"})
    )


def build_position_audit(values, team):
    baselines = positional_baselines(values)
    team_df = add_audit_values(values[values["Team Name"].eq(team)])
    lineup = active_player_rows(team_df)
    position_summary = (
        lineup.groupby("Position")
        .agg(
            Filled=("Name", lambda names: names.astype(str).str.len().gt(0).sum()),
            Slots=("Name", "count"),
            Team_Value=("Audit_Value", "sum"),
            Avg_Player_Value=("Audit_Value", "mean"),
            Raw_Team_Value=("Current_Value", "sum"),
            Team_Surplus=("Audit_Surplus", "sum"),
        )
        .reset_index()
    )
    position_summary = position_summary.merge(baselines, on="Position", how="left")
    position_summary["Expected_Position_Value"] = (
        position_summary["League_Avg_Value"].fillna(0) * position_summary["Slots"]
    )
    position_summary["Value_Gap"] = position_summary["Team_Value"] - position_summary["Expected_Position_Value"]
    position_summary["Need_Level"] = pd.cut(
        position_summary["Value_Gap"],
        bins=[-999, -12, -5, 5, 999],
        labels=["Major need", "Need", "Neutral", "Strength"],
    ).astype(str)
    position_summary["Sort"] = position_summary["Position"].apply(slot_sort_key)
    return position_summary.sort_values(["Sort", "Value_Gap"]).drop(columns=["Sort"])


def build_category_audit(values, team):
    present = {}
    for category, candidates in CATEGORY_COLUMNS.items():
        for col in candidates:
            if col in values.columns:
                present[category] = col
                break
    if not present:
        return pd.DataFrame([{
            "Category": "Not available",
            "Team_Total": pd.NA,
            "League_Avg": pd.NA,
            "Gap": pd.NA,
            "Need_Level": "Category projections not loaded yet",
        }])

    active = values[values["Active_Slot"].ne("Reserve")]
    rows = []
    for category, col in present.items():
        by_team = active.groupby("Team Name")[col].sum()
        team_total = by_team.get(team, 0)
        league_avg = by_team.mean()
        gap = team_total - league_avg
        rows.append({
            "Category": category,
            "Team_Total": team_total,
            "League_Avg": league_avg,
            "Gap": gap,
            "Need_Level": "Need" if gap < 0 else "Strength",
        })
    return pd.DataFrame(rows).sort_values("Gap")


def player_positions_set(row):
    return {pos.strip().upper() for pos in str(row.get("Positions", "")).split("/") if pos.strip()}


def position_matches(row, position):
    positions = player_positions_set(row)
    if position in positions:
        return True
    if position == "MI" and positions.intersection({"2B", "SS", "MI"}):
        return True
    if position == "CI" and positions.intersection({"1B", "3B", "CI"}):
        return True
    if position == "UTIL" and positions.intersection({"C", "1B", "2B", "3B", "SS", "OF", "DH", "UTIL"}):
        return True
    return False


def trade_chips(values, team, position_summary):
    team_df = add_audit_values(values[values["Team Name"].eq(team)])
    strengths = set(position_summary[position_summary["Need_Level"].eq("Strength")]["Position"])
    rows = []
    for _, row in team_df.iterrows():
        is_reserve = row["Active_Slot"] == "Reserve"
        is_strength_pos = any(position_matches(row, position) for position in strengths)
        is_good_prospect = bool(row.get("Is_Prospect")) and float(row.get("Audit_Value", 0) or 0) >= 3
        is_surplus = float(row.get("Audit_Surplus", 0) or 0) >= 3
        if is_reserve or is_strength_pos or is_good_prospect or is_surplus:
            reason = []
            if is_reserve:
                reason.append("reserve")
            if is_strength_pos:
                reason.append("position strength")
            if is_good_prospect:
                reason.append("prospect value")
            if is_surplus:
                reason.append("salary surplus")
            rows.append({
                "Name": row["Name"],
                "Positions": row["Positions"],
                "Salary": row["Salary"],
                "Current_Value": row["Current_Value"],
                "Audit_Value": row["Audit_Value"],
                "Current_Surplus": row["Current_Surplus"],
                "Audit_Surplus": row["Audit_Surplus"],
                "Is_Prospect": row["Is_Prospect"],
                "Trade_Reason": ", ".join(reason),
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["Audit_Value", "Audit_Surplus"], ascending=False)


def target_players(values, team, needs):
    other = add_audit_values(values[~values["Team Name"].eq(team)])
    targets = []
    for _, need in needs.iterrows():
        position = need["Position"]
        candidates = other[other.apply(lambda row: position_matches(row, position), axis=1)].copy()
        candidates = candidates[candidates["Audit_Value"] >= max(5, float(need.get("Avg_Player_Value", 0) or 0) + 2)]
        candidates["Need_Gap_Abs"] = (candidates["Audit_Value"] - max(float(need.get("Avg_Player_Value", 0) or 0), 0)).abs()
        for _, row in candidates.sort_values(["Need_Gap_Abs", "Audit_Value"], ascending=[True, False]).head(16).iterrows():
            targets.append({
                "Need": position,
                "Target_Team": row["Team Name"],
                "Target": row["Name"],
                "Target_Positions": row["Positions"],
                "Target_Salary": row["Salary"],
                "Target_Value": row["Current_Value"],
                "Target_Audit_Value": row["Audit_Value"],
                "Target_Surplus": row["Current_Surplus"],
                "Target_Audit_Surplus": row["Audit_Surplus"],
            })
    return pd.DataFrame(targets)


def build_trade_packages(chips, target_value, max_packages=8):
    chips = chips.copy()
    chips = chips[chips["Audit_Value"] > 0].head(14)
    packages = []
    for _, chip in chips.iterrows():
        total = float(chip["Audit_Value"])
        if target_value * 0.7 <= total <= target_value * 1.25:
            packages.append(([chip], total, "1-for-1"))
    for first_idx, first in chips.iterrows():
        for second_idx, second in chips.iterrows():
            if second_idx <= first_idx:
                continue
            total = float(first["Audit_Value"]) + float(second["Audit_Value"])
            if target_value * 0.8 <= total <= target_value * 1.35:
                packages.append(([first, second], total, "2-for-1"))
    packages.sort(key=lambda package: (abs(package[1] - target_value), len(package[0])))
    return packages[:max_packages]


def trade_ideas(values, team, position_summary, max_rows=25):
    needs = position_summary[position_summary["Need_Level"].isin(["Major need", "Need"])].sort_values("Value_Gap")
    chips = trade_chips(values, team, position_summary)
    targets = target_players(values, team, needs)
    if chips.empty or targets.empty:
        return pd.DataFrame(columns=[
            "Trade_From", "Trade_Chip", "Trade_To", "Target", "Need", "Idea_Type", "Rationale",
        ])

    ideas = []
    for _, target in targets.iterrows():
        target_audit_value = float(target["Target_Audit_Value"])
        if target_audit_value >= SUPERSTAR_VALUE:
            superstar_chip_available = chips["Audit_Value"].ge(target_audit_value * 0.8).any()
            if not superstar_chip_available:
                continue
        packages = build_trade_packages(chips, target_audit_value)
        for package, package_value, package_type in packages:
            chip_names = " + ".join(chip["Name"] for chip in package)
            chip_positions = " + ".join(chip["Positions"] for chip in package)
            chip_reasons = "; ".join(chip["Trade_Reason"] for chip in package)
            has_prospect = any(bool(chip["Is_Prospect"]) for chip in package)
            idea_type = "Future-for-now" if has_prospect else "Reallocate surplus"
            if target["Target_Audit_Surplus"] >= 8:
                idea_type = "Buy elite surplus"
            ideas.append({
                "Trade_From": team,
                "Package_Type": package_type,
                "Trade_Chip": chip_names,
                "Chip_Positions": chip_positions,
                "Chip_Value": round(package_value, 2),
                "Chip_Reason": chip_reasons,
                "Trade_To": target["Target_Team"],
                "Target": target["Target"],
                "Target_Positions": target["Target_Positions"],
                "Target_Value": target["Target_Value"],
                "Target_Audit_Value": target["Target_Audit_Value"],
                "Need": target["Need"],
                "Idea_Type": idea_type,
                "Rationale": f"Move {chip_reasons} to improve {target['Need']}.",
            })
    if not ideas:
        return pd.DataFrame()
    ideas_df = pd.DataFrame(ideas).sort_values(
        ["Need", "Package_Type", "Target_Audit_Value"],
        ascending=[True, True, False],
    )
    one_for_one = ideas_df[ideas_df["Package_Type"].eq("1-for-1")].head(max_rows // 2)
    packages = ideas_df[~ideas_df["Package_Type"].eq("1-for-1")].head(max_rows - len(one_for_one))
    if len(packages) < max_rows - len(one_for_one):
        remaining = ideas_df.drop(index=one_for_one.index.union(packages.index)).head(max_rows - len(one_for_one) - len(packages))
        packages = pd.concat([packages, remaining], ignore_index=False)
    return pd.concat([one_for_one, packages], ignore_index=True).head(max_rows)


def build_roster_audit(values, team):
    resolved_team = resolve_team_name(values, team)
    position_summary = build_position_audit(values, resolved_team)
    category_summary = build_category_audit(values, resolved_team)
    chips = trade_chips(values, resolved_team, position_summary)
    ideas = trade_ideas(values, resolved_team, position_summary)
    team_df = add_audit_values(values[values["Team Name"].eq(resolved_team)])
    overview = pd.DataFrame([{
        "Team": resolved_team,
        "Roster_Size": len(team_df),
        "Salary": team_df["Salary"].sum(),
        "Current_Value": team_df["Audit_Value"].sum(),
        "Current_Surplus": team_df["Audit_Surplus"].sum(),
        "Raw_Current_Value": team_df["Current_Value"].sum(),
        "Active_Value": team_df[team_df["Active_Slot"].ne("Reserve")]["Audit_Value"].sum(),
        "Reserve_Value": team_df[team_df["Active_Slot"].eq("Reserve")]["Audit_Value"].sum(),
        "Prospects": int(team_df["Is_Prospect"].sum()),
        "Major_Needs": int(position_summary["Need_Level"].eq("Major need").sum()),
        "Needs": int(position_summary["Need_Level"].isin(["Major need", "Need"]).sum()),
        "Strengths": int(position_summary["Need_Level"].eq("Strength").sum()),
    }])
    return {
        "team": resolved_team,
        "overview": overview,
        "positions": position_summary,
        "categories": category_summary,
        "trade_chips": chips,
        "trade_ideas": ideas,
    }


def main():
    parser = argparse.ArgumentParser(description="Audit a roster's positional strengths, weaknesses, and trade directions.")
    parser.add_argument("team")
    parser.add_argument("--rosters", default="current_rosters.csv")
    parser.add_argument("--hitters", default="hitters_ros.csv")
    parser.add_argument("--pitchers", default="pitchers_ros.csv")
    parser.add_argument("--avg", default="fgpts_avgvalues.csv")
    parser.add_argument("--prospects", default=os.path.join("data", "Baseball Composite Prospect List 2026 - List.csv"))
    parser.add_argument("--data-source", choices=["static", "cache"], default=None)
    args = parser.parse_args()

    if args.data_source:
        paths = resolve_data_paths(args.data_source)
        args.rosters = paths.rosters
        args.hitters = paths.hitters_ros
        args.pitchers = paths.pitchers_ros
        args.avg = paths.avg_values
        args.prospects = paths.prospects

    values, _ = build_player_value_table(args.rosters, args.hitters, args.pitchers, args.avg, args.prospects)
    audit = build_roster_audit(values, args.team)
    print("\nOverview")
    print(audit["overview"].to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print("\nPositional Audit")
    print(audit["positions"].to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print("\nTrade Ideas")
    print(audit["trade_ideas"].head(15).to_string(index=False, float_format=lambda value: f"{value:.2f}"))


if __name__ == "__main__":
    main()
