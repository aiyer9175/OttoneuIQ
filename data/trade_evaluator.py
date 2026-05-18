import argparse
import itertools
import urllib.request
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Pandas requires version")
import pandas as pd

from data_sources import resolve_data_paths
from post_auction import load_average_values, load_rosters, normalized_player_id, resolve_team_name
from value_engine import build_player_value_table, load_mlb_status, load_or_build_mlb_stock, resolve_player


SCARCE_POSITIONS = {"C", "SS", "SP", "RP"}
OTTONEU_ROSTER_EXPORT_URL = "https://ottoneu.fangraphs.com/{league_id}/rosterexport?csv=1"
SALARY_CAP = 400
ROSTER_LIMIT = 40


def _num(value, default=0.0):
    if pd.isna(value):
        return default
    return float(value)


def _positions_set(value):
    return {part.strip().upper() for part in str(value or "").replace(",", "/").split("/") if part.strip()}


def _team_salary(values, team):
    return float(pd.to_numeric(values[values["Team Name"].eq(team)]["Salary"], errors="coerce").fillna(0).sum())


def _max_legal_salary(roster_size):
    return SALARY_CAP - max(0, ROSTER_LIMIT - int(roster_size))


def _team_position_profile(team_df):
    profile = {}
    if "Active_Slot" not in team_df.columns:
        return profile
    active = team_df[team_df["Active_Slot"].ne("Reserve")].copy()
    active["Current_Value"] = pd.to_numeric(active["Current_Value"], errors="coerce").fillna(0)
    for position, group in active.groupby("Active_Slot"):
        count = len(group)
        total = float(group["Current_Value"].sum())
        avg = total / count if count else 0.0
        if count == 0 or avg < 4:
            level = "need"
        elif avg >= 12:
            level = "strength"
        else:
            level = "neutral"
        profile[str(position)] = {"count": count, "total": total, "avg": avg, "level": level}
    return profile


def _matching_positions(positions, profile):
    matches = set()
    for position in positions:
        if position in profile:
            matches.add(position)
        if position in {"2B", "SS"} and "MI" in profile:
            matches.add("MI")
        if position in {"1B", "3B"} and "CI" in profile:
            matches.add("CI")
        if position in {"C", "1B", "2B", "3B", "SS", "OF", "DH", "UTIL"} and "UTIL" in profile:
            matches.add("UTIL")
    return matches


def _opponent_context(values, opponent_team, target_rows, package_rows):
    if not opponent_team or opponent_team == "Multiple":
        return {"score_adjustment": 0.0, "notes": "opponent context unavailable for multi-team target"}

    opponent_df = values[values["Team Name"].eq(opponent_team)].copy()
    roster_size = len(opponent_df)
    salary = _team_salary(values, opponent_team)
    target_salary = float(pd.to_numeric(target_rows["Salary"], errors="coerce").fillna(0).sum())
    package_salary = float(pd.to_numeric(package_rows["Salary"], errors="coerce").fillna(0).sum())
    target_surplus = float(pd.to_numeric(target_rows["Current_Surplus"], errors="coerce").fillna(0).sum())
    package_surplus = float(pd.to_numeric(package_rows["Current_Surplus"], errors="coerce").fillna(0).sum())
    target_count = len(target_rows)
    package_count = len(package_rows)
    salary_delta = package_salary - target_salary
    roster_delta = package_count - target_count
    post_roster_size = roster_size + roster_delta
    post_salary = salary + salary_delta
    max_salary = _max_legal_salary(post_roster_size)

    profile = _team_position_profile(opponent_df)
    package_positions = set().union(*package_rows["Positions"].apply(_positions_set).tolist())
    target_positions = set().union(*target_rows["Positions"].apply(_positions_set).tolist())
    package_matches = _matching_positions(package_positions, profile)
    target_matches = _matching_positions(target_positions, profile)
    need_matches = [pos for pos in package_matches if profile.get(pos, {}).get("level") == "need"]
    strength_targets = [pos for pos in target_matches if profile.get(pos, {}).get("level") == "strength"]
    need_targets = [pos for pos in target_matches if profile.get(pos, {}).get("level") == "need"]

    adjustment = 0.0
    notes = []
    if need_matches:
        adjustment -= 1.4 * len(need_matches)
        notes.append(f"fills opponent need: {', '.join(sorted(need_matches))}")
    if need_targets:
        adjustment += 1.8 * len(need_targets)
        notes.append(f"asks opponent to lose need position: {', '.join(sorted(need_targets))}")
    if strength_targets:
        adjustment -= 0.8 * len(strength_targets)
        notes.append(f"targets opponent strength: {', '.join(sorted(strength_targets))}")

    opponent_surplus_delta = package_surplus - target_surplus
    if opponent_surplus_delta > 0:
        adjustment -= min(opponent_surplus_delta, 8) * 0.20
        notes.append(f"opponent gains surplus {opponent_surplus_delta:+.1f}")
    elif opponent_surplus_delta < -3:
        adjustment += min(abs(opponent_surplus_delta), 10) * 0.25
        notes.append(f"opponent loses surplus {opponent_surplus_delta:+.1f}")

    if salary >= 380 and salary_delta < 0:
        adjustment -= min(abs(salary_delta), 15) * 0.18
        notes.append(f"salary relief ${abs(salary_delta):.0f}")
    elif salary_delta > 0 and post_salary > max_salary:
        adjustment += 8.0
        notes.append("would violate cap/empty-slot reserve")
    elif salary_delta > 8:
        adjustment += 1.5
        notes.append(f"adds salary ${salary_delta:.0f}")

    if post_roster_size > ROSTER_LIMIT:
        adjustment += 5.0
        notes.append("would exceed 40-player roster")
    elif roster_delta < 0:
        adjustment -= 0.4
        notes.append("opens roster spot")

    return {
        "score_adjustment": adjustment,
        "notes": "; ".join(notes) if notes else "neutral opponent fit",
        "opponent_salary_delta": salary_delta,
        "opponent_surplus_delta": opponent_surplus_delta,
        "opponent_post_salary": post_salary,
        "opponent_post_roster": post_roster_size,
    }


def parse_players(value):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def fetch_ottoneu_roster_export(league_id, destination, opener=None):
    clean_id = str(league_id or "").strip()
    if not clean_id.isdigit():
        raise ValueError("League number must be numeric.")
    opener = opener or urllib.request.urlopen
    url = OTTONEU_ROSTER_EXPORT_URL.format(league_id=clean_id)
    request = urllib.request.Request(url, headers={"User-Agent": "OttoneuIQ roster import"})
    with opener(request, timeout=60) as response:
        content = response.read()
    destination = Path(destination)
    destination.write_bytes(content)
    return str(destination)


def team_context(values, team):
    team_df = values[values["Team Name"].eq(team)]
    return {
        "salary": float(team_df["Salary"].sum()),
        "value": float(team_df["Current_Value"].sum()),
        "surplus": float(team_df["Current_Surplus"].sum()),
        "roster_size": int(len(team_df)),
    }


def player_rows(values, players, expected_team=None):
    rows = []
    for player in players:
        row = resolve_player(values, player, team=expected_team)
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=values.columns)
    return pd.DataFrame(rows)


def package_summary(rows):
    if rows.empty:
        return {"Salary": 0.0, "Value": 0.0, "Surplus": 0.0, "Stock_Change": 0.0}
    return {
        "Salary": float(rows["Salary"].sum()),
        "Value": float(rows["Current_Value"].sum()),
        "Surplus": float(rows["Current_Surplus"].sum()),
        "Stock_Change": float(rows["Stock_Change"].sum()),
    }


def evaluate_trade(values, team_a, sends, receives, team_b=None):
    team_a = resolve_team_name(values.rename(columns={"Team Name": "Team Name"}), team_a)
    if team_b:
        team_b = resolve_team_name(values.rename(columns={"Team Name": "Team Name"}), team_b)

    outgoing = player_rows(values, sends, expected_team=team_a)
    incoming = player_rows(values, receives, expected_team=team_b)
    out_summary = package_summary(outgoing)
    in_summary = package_summary(incoming)

    context = team_context(values, team_a)
    salary_delta = in_summary["Salary"] - out_summary["Salary"]
    value_delta = in_summary["Value"] - out_summary["Value"]
    surplus_delta = in_summary["Surplus"] - out_summary["Surplus"]
    post_salary = context["salary"] + salary_delta
    post_value = context["value"] + value_delta
    post_surplus = context["surplus"] + surplus_delta

    verdict = "ACCEPT"
    if surplus_delta < -5 and value_delta < -3:
        verdict = "DECLINE"
    elif surplus_delta < -2:
        verdict = "LEAN DECLINE"
    elif surplus_delta < 2:
        verdict = "FAIR"
    elif surplus_delta < 6:
        verdict = "LEAN ACCEPT"

    return {
        "team": team_a,
        "team_b": team_b,
        "outgoing": outgoing,
        "incoming": incoming,
        "out_summary": out_summary,
        "in_summary": in_summary,
        "salary_delta": salary_delta,
        "value_delta": value_delta,
        "surplus_delta": surplus_delta,
        "post_salary": post_salary,
        "post_value": post_value,
        "post_surplus": post_surplus,
        "verdict": verdict,
    }


def _package_display(rows):
    return " + ".join(
        f"{row['Name']} ({row.get('Positions', '')})"
        for _, row in rows.iterrows()
    )


def _package_tier(rows):
    max_value = pd.to_numeric(rows["Current_Value"], errors="coerce").max()
    if max_value >= 25:
        return "includes star"
    if max_value >= 12:
        return "core piece"
    return "peripheral/depth"


def recommend_trade_packages(
    values,
    team,
    target_player,
    target_team=None,
    max_package_size=3,
    limit=12,
    include_negative_surplus=False,
):
    team = resolve_team_name(values.rename(columns={"Team Name": "Team Name"}), team)
    if target_team:
        target_team = resolve_team_name(values.rename(columns={"Team Name": "Team Name"}), target_team)
    target_names = target_player if isinstance(target_player, list) else parse_players(target_player)
    if not target_names and target_player:
        target_names = [target_player]
    if not target_names:
        raise ValueError("Enter at least one target player.")
    target_rows = player_rows(values, target_names, expected_team=target_team)
    own_targets = target_rows[target_rows["Team Name"].eq(team)]
    if not own_targets.empty:
        names = ", ".join(own_targets["Name"].astype(str).tolist())
        raise ValueError(f"Target package includes player(s) already on {team}: {names}.")

    roster = values[values["Team Name"].eq(team)].copy()
    roster["Current_Value"] = pd.to_numeric(roster["Current_Value"], errors="coerce").fillna(0)
    roster["Current_Surplus"] = pd.to_numeric(roster["Current_Surplus"], errors="coerce").fillna(0)
    roster["Salary"] = pd.to_numeric(roster["Salary"], errors="coerce").fillna(0)
    if not include_negative_surplus:
        roster = roster[(roster["Current_Surplus"] >= -8) | (roster["Current_Value"] >= 12)]
    roster = roster[roster["Current_Value"] > 0].sort_values("Current_Value", ascending=False)

    target_value = float(pd.to_numeric(target_rows["Current_Value"], errors="coerce").fillna(0).sum())
    target_surplus = float(pd.to_numeric(target_rows["Current_Surplus"], errors="coerce").fillna(0).sum())
    target_salary = float(pd.to_numeric(target_rows["Salary"], errors="coerce").fillna(0).sum())
    target_positions = set().union(*target_rows["Positions"].apply(_positions_set).tolist())
    target_display = _package_display(target_rows)
    target_team_display = (
        target_rows["Team Name"].iloc[0]
        if target_rows["Team Name"].nunique(dropna=False) == 1
        else "Multiple"
    )

    rows = []
    max_package_size = max(1, min(int(max_package_size), 4))
    for size in range(1, max_package_size + 1):
        for combo in itertools.combinations(roster.index.tolist(), size):
            package = roster.loc[list(combo)]
            outgoing_value = float(package["Current_Value"].sum())
            outgoing_surplus = float(package["Current_Surplus"].sum())
            outgoing_salary = float(package["Salary"].sum())
            value_delta = target_value - outgoing_value
            surplus_delta = target_surplus - outgoing_surplus
            salary_delta = target_salary - outgoing_salary
            if target_value >= 8 and outgoing_value < target_value * 0.55:
                continue
            if outgoing_value > max(target_value * 1.55, target_value + 10):
                continue

            max_piece_value = float(package["Current_Value"].max())
            superstar_penalty = max(0.0, max_piece_value - max(target_value * 1.15, 18.0)) * 0.45
            roster_spot_penalty = max(0, size - 1) * 0.75
            surplus_penalty = abs(surplus_delta) * 0.45
            value_penalty = abs(value_delta) * 0.90
            if value_delta < -4:
                value_penalty += abs(value_delta + 4) * 0.70
            if value_delta > 5:
                value_penalty += (value_delta - 5) * 0.40
            position_bonus = 0.0
            package_positions = set().union(*package["Positions"].apply(_positions_set).tolist())
            if target_positions & SCARCE_POSITIONS and not package_positions.intersection(target_positions):
                position_bonus -= 0.6
            opponent = _opponent_context(values, target_team_display, target_rows, package)
            score = (
                value_penalty + surplus_penalty + superstar_penalty + roster_spot_penalty
                + position_bonus + opponent.get("score_adjustment", 0.0)
            )
            rows.append({
                "Target": target_display,
                "Target_Team": target_team_display,
                "Package": _package_display(package),
                "Package_Size": size,
                "Package_Tier": _package_tier(package),
                "Outgoing_Salary": outgoing_salary,
                "Outgoing_Value": outgoing_value,
                "Outgoing_Surplus": outgoing_surplus,
                "Receive_Salary": target_salary,
                "Receive_Value": target_value,
                "Receive_Surplus": target_surplus,
                "Value_Delta": value_delta,
                "Surplus_Delta": surplus_delta,
                "Salary_Delta": salary_delta,
                "Opponent_Salary_Delta": opponent.get("opponent_salary_delta", pd.NA),
                "Opponent_Surplus_Delta": opponent.get("opponent_surplus_delta", pd.NA),
                "Opponent_Post_Salary": opponent.get("opponent_post_salary", pd.NA),
                "Opponent_Post_Roster": opponent.get("opponent_post_roster", pd.NA),
                "Opponent_Context": opponent.get("notes", ""),
                "Fit_Score": score,
            })
    if not rows:
        return pd.DataFrame()
    recommendations = pd.DataFrame(rows).sort_values(
        ["Fit_Score", "Package_Size", "Value_Delta"],
        ascending=[True, True, False],
    )
    return recommendations.head(int(limit)).reset_index(drop=True)


def _load_ros_pool(hitters, pitchers):
    frames = []
    for path, player_type, playing_time_col in [
        (hitters, "hitter", "PA"),
        (pitchers, "pitcher", "IP"),
    ]:
        df = pd.read_csv(path)
        df["PlayerIdKey"] = df["PlayerId"].apply(normalized_player_id)
        df["MLBAMIDKey"] = df["MLBAMID"].apply(normalized_player_id)
        df["Player_Type"] = player_type
        df["Positions"] = df["POS"].fillna("").astype(str)
        df["Current_Value"] = pd.to_numeric(df["Dollars"], errors="coerce").fillna(0)
        df["Projected_Points"] = pd.to_numeric(df["rPTS"], errors="coerce").fillna(0)
        df["Projected_Playing_Time"] = pd.to_numeric(df[playing_time_col], errors="coerce").fillna(0)
        frames.append(df[[
            "Name", "Team", "PlayerIdKey", "MLBAMIDKey", "Player_Type", "Positions",
            "Current_Value", "Projected_Points", "Projected_Playing_Time",
        ]])
    pool = pd.concat(frames, ignore_index=True)
    pool = pool[pool["PlayerIdKey"].notna()]
    return pool.sort_values("Current_Value", ascending=False).drop_duplicates("PlayerIdKey", keep="first")


def build_waiver_wire_board(
    rosters,
    hitters,
    pitchers,
    avg,
    mlb_stock=None,
    mlb_status=None,
    min_value=-5,
    limit=50,
):
    roster = load_rosters(rosters)
    rostered_ids = set(roster["PlayerIdKey"].dropna().astype(str))
    pool = _load_ros_pool(hitters, pitchers)
    available = pool[~pool["PlayerIdKey"].isin(rostered_ids)].copy()

    avg_values = load_average_values(avg)
    avg_values["PlayerIdKey"] = avg_values["PlayerIdKey"].apply(normalized_player_id)
    avg_values = avg_values[avg_values["PlayerIdKey"].notna()].drop_duplicates("PlayerIdKey")
    available = available.merge(
        avg_values[["PlayerIdKey", "Avg_Salary", "Last10_Salary", "Roster%"]],
        on="PlayerIdKey",
        how="left",
    )

    stock = load_or_build_mlb_stock(mlb_stock)
    stock_cols = [
        "PlayerIdKey", "MLB_Stock_Change", "YTD_Value", "YTD_ROS_Gap",
        "Skill_Score", "Role_Change", "Stock_Label", "Confidence_Label",
    ]
    for col in stock_cols:
        if col not in stock.columns:
            stock[col] = pd.NA
    stock["PlayerIdKey"] = stock["PlayerIdKey"].apply(normalized_player_id)
    stock = stock[stock["PlayerIdKey"].notna()].drop_duplicates("PlayerIdKey")
    available = available.merge(stock[stock_cols], on="PlayerIdKey", how="left")

    status = load_mlb_status(mlb_status)
    if not status.empty:
        status = status[status["MLBAMIDKey"].notna()].drop_duplicates("MLBAMIDKey")
        available = available.merge(status, on="MLBAMIDKey", how="left")

    available["Salary_Assumption"] = 1.0
    available["Available_Surplus"] = available["Current_Value"] - available["Salary_Assumption"]
    available["MLB_Stock_Change"] = pd.to_numeric(available["MLB_Stock_Change"], errors="coerce").fillna(0)
    available["YTD_ROS_Gap"] = pd.to_numeric(available["YTD_ROS_Gap"], errors="coerce").fillna(0)
    available["Roster%"] = pd.to_numeric(available["Roster%"], errors="coerce").fillna(0)
    available["Skill_Score"] = pd.to_numeric(available["Skill_Score"], errors="coerce").fillna(0.5)
    available["Status_Flag"] = available.get("Status_Flag", pd.Series(index=available.index, dtype=object)).fillna("UNKNOWN")
    scarcity = available["Positions"].apply(lambda value: 1.25 if _positions_set(value) & SCARCE_POSITIONS else 0.0)
    status_penalty = available["Status_Flag"].map({"SENT_DOWN": -4.0, "IL": -5.0, "DFA": -6.0, "RELEASED": -8.0}).fillna(0)
    available["Composite_Score"] = (
        available["Current_Value"]
        + available["Available_Surplus"].clip(lower=0) * 0.35
        + available["MLB_Stock_Change"] * 0.50
        + available["YTD_ROS_Gap"].clip(lower=0, upper=20) * 0.08
        + (available["Skill_Score"] - 0.5) * 3.0
        + available["Roster%"].clip(upper=100) * 0.02
        + scarcity
        + status_penalty
    )
    available["Player"] = available.apply(lambda row: f"{row['Name']} ({row['Positions']})", axis=1)
    available = available[pd.to_numeric(available["Current_Value"], errors="coerce") >= float(min_value)]
    columns = [
        "Player", "Name", "Team", "Positions", "Player_Type", "Current_Value",
        "Available_Surplus", "Composite_Score", "MLB_Stock_Change", "YTD_Value",
        "YTD_ROS_Gap", "Skill_Score", "Roster%", "Avg_Salary", "Last10_Salary",
        "Stock_Label", "Role_Change", "Confidence_Label", "MLB_Status",
        "Status_Flag", "Latest_Transaction_Date",
    ]
    for col in columns:
        if col not in available.columns:
            available[col] = pd.NA
    return available.sort_values("Composite_Score", ascending=False).head(int(limit))[columns].reset_index(drop=True)


def print_package(label, rows):
    cols = [
        "Team Name", "Name", "Positions", "Salary", "Current_Value", "Current_Surplus",
        "Stock_Change", "YTD_Value", "Stock_Label", "Role_Change", "Value_Source",
    ]
    cols = [col for col in cols if col in rows.columns]
    print(f"\n{label}")
    if rows.empty:
        print("None")
        return
    print(rows[cols].to_string(index=False, float_format=lambda value: f"{value:.2f}"))


def print_evaluation(result):
    print(f"Trade verdict for {result['team']}: {result['verdict']}")
    print_package("Outgoing", result["outgoing"])
    print_package("Incoming", result["incoming"])
    print("\nSummary")
    print(f"Value delta:   {result['value_delta']:+.2f}")
    print(f"Salary delta:  ${result['salary_delta']:+.2f}")
    print(f"Surplus delta: {result['surplus_delta']:+.2f}")
    print(f"Post-trade salary: ${result['post_salary']:.2f}")
    print(f"Post-trade roster surplus: {result['post_surplus']:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Ottoneu trade evaluator.")
    parser.add_argument("--team", required=True, help="Team receiving the incoming package.")
    parser.add_argument("--send", required=True, help="Comma-separated players this team sends.")
    parser.add_argument("--receive", required=True, help="Comma-separated players this team receives.")
    parser.add_argument("--from-team", default=None, help="Optional team currently rostering incoming players.")
    parser.add_argument("--rosters", default="current_rosters.csv")
    parser.add_argument("--hitters", default="hitters_ros.csv")
    parser.add_argument("--pitchers", default="pitchers_ros.csv")
    parser.add_argument("--avg", default="fgpts_avgvalues.csv")
    parser.add_argument("--prospects", default="data/Baseball Composite Prospect List 2026 - List.csv")
    parser.add_argument("--prospect-updates", default="prospect_value_updates.csv")
    parser.add_argument("--data-source", choices=["static", "cache"], default=None)
    args = parser.parse_args()

    if args.data_source:
        paths = resolve_data_paths(args.data_source)
        args.rosters = paths.rosters
        args.hitters = paths.hitters_ros
        args.pitchers = paths.pitchers_ros
        args.avg = paths.avg_values
        args.prospects = paths.prospects

    values, _ = build_player_value_table(
        args.rosters, args.hitters, args.pitchers, args.avg, args.prospects, args.prospect_updates
    )
    result = evaluate_trade(
        values,
        team_a=args.team,
        sends=parse_players(args.send),
        receives=parse_players(args.receive),
        team_b=args.from_team,
    )
    print_evaluation(result)


if __name__ == "__main__":
    main()
