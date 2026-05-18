import argparse
import os
import warnings

warnings.filterwarnings("ignore", message="Pandas requires version")
import pandas as pd

from mlb_stock import DEFAULT_OUTPUT as MLB_STOCK_OUTPUT, build_mlb_stock
from post_auction import build_reports, normalize_name
from prospect_updates import DEFAULT_OUTPUT, build_prospect_updates, normalize_text
from data_sources import resolve_data_paths
from prospect_status import add_recent_mlb_playing_time, apply_prospect_graduation
from role_risk import apply_role_risk_adjustment


DEFAULT_ROSTERS = "current_rosters.csv"
DEFAULT_HITTERS = "hitters_ros.csv"
DEFAULT_PITCHERS = "pitchers_ros.csv"
DEFAULT_AVG = "fgpts_avgvalues.csv"
DEFAULT_PROSPECTS = os.path.join("data", "Baseball Composite Prospect List 2026 - List.csv")
DEFAULT_MLB_STATUS = "mlb_player_status.csv"


def normalized_player_id(value):
    if pd.isna(value):
        return None
    try:
        return str(int(float(value)))
    except ValueError:
        clean = str(value).strip()
        return clean or None


def load_or_build_prospect_updates(path=DEFAULT_OUTPUT):
    if path and os.path.exists(path):
        return pd.read_csv(path)
    return build_prospect_updates()


def load_or_build_mlb_stock(path=MLB_STOCK_OUTPUT):
    if path and os.path.exists(path):
        return pd.read_csv(path)
    return build_mlb_stock()


def load_mlb_status(path=DEFAULT_MLB_STATUS):
    columns = [
        "MLBAMIDKey", "MLB_Status", "Status_Flag", "Current_Team",
        "Latest_Transaction_Date", "Latest_Transaction_Type",
        "Latest_Transaction_Description", "FetchedAtUTC",
    ]
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=columns)
    status = pd.read_csv(path)
    for col in columns:
        if col not in status.columns:
            status[col] = pd.NA
    status["MLBAMIDKey"] = status["MLBAMIDKey"].apply(normalized_player_id)
    return status[columns]


def build_player_value_table(
    rosters=DEFAULT_ROSTERS,
    hitters=DEFAULT_HITTERS,
    pitchers=DEFAULT_PITCHERS,
    avg=DEFAULT_AVG,
    prospects=DEFAULT_PROSPECTS,
    prospect_updates=DEFAULT_OUTPUT,
    mlb_stock=MLB_STOCK_OUTPUT,
    mlb_status=DEFAULT_MLB_STATUS,
):
    report, summary = build_reports(rosters, hitters, pitchers, avg, prospects, mlb_stock)
    values = report.copy()
    values["NameKey"] = values["Name"].apply(normalize_text)

    updates = load_or_build_prospect_updates(prospect_updates)
    updates["NameKey"] = updates["Name"].apply(normalize_text)
    update_cols = [
        "NameKey", "Updated_Prospect_Value", "Prospect_Update", "Confidence_Label",
        "Evidence_Score", "Pipeline_Rank", "Composite_Rank", "Match_Status",
    ]
    updates = updates.sort_values("Updated_Prospect_Value", ascending=False).drop_duplicates("NameKey")
    values = values.merge(updates[update_cols], on="NameKey", how="left")
    values["Has_Minors_Data"] = values["Updated_Prospect_Value"].notna()
    prospect_source_match = (
        values["Prospect_Listed"].fillna(False)
        | values["Pipeline_Rank"].notna()
        | values["Composite_Rank"].notna()
    )
    values["Prospect_Listed"] = prospect_source_match
    values["Is_Prospect"] = prospect_source_match
    prospect_update_cols = [
        "Updated_Prospect_Value", "Prospect_Update", "Confidence_Label",
        "Evidence_Score", "Pipeline_Rank", "Composite_Rank", "Match_Status",
    ]
    non_listed_minors = values["Has_Minors_Data"] & ~values["Prospect_Listed"]
    values.loc[non_listed_minors, prospect_update_cols] = pd.NA

    stock = load_or_build_mlb_stock(mlb_stock)
    stock_cols = [
        "PlayerIdKey", "MLBAMIDKey", "Preseason_Value", "ROS_Value", "Projection_Change",
        "MLB_Stock_Change", "Skill_Score", "Role_Change", "Stock_Label",
        "YTD_Value", "YTD_ROS_Gap", "Banked_Value_Signal", "Confidence_Label",
        "Player_Type", "YTD_PA", "YTD_IP", "ROS_IP",
    ]
    for col in stock_cols:
        if col not in stock.columns:
            stock[col] = pd.NA
    stock = stock[stock["PlayerIdKey"].notna()].drop_duplicates("PlayerIdKey")
    values = values.merge(
        stock[stock_cols],
        on="PlayerIdKey",
        how="left",
        suffixes=("", "_MLB"),
    )
    values = add_recent_mlb_playing_time(values)
    values = apply_prospect_graduation(values)

    values["Base_Value"] = values["Future_Value"]
    values["Current_Value"] = values["Future_Value"]
    mlb_match = values["ROS_Value"].notna()
    values.loc[mlb_match, "Base_Value"] = values.loc[mlb_match, "Preseason_Value"]
    values.loc[mlb_match, "Current_Value"] = values.loc[mlb_match, "ROS_Value"]
    values["Role_Risk_Adjustment"] = 0.0
    adjusted = apply_role_risk_adjustment(values.loc[mlb_match], "Current_Value")
    for col in ["Current_Value", "Role_Risk_Adjustment"]:
        values.loc[mlb_match, col] = adjusted[col]
    prospect_match = values["Updated_Prospect_Value"].notna() & values["Is_Prospect"] & ~mlb_match
    values.loc[prospect_match, "Current_Value"] = values.loc[prospect_match, "Updated_Prospect_Value"]
    values["Current_Surplus"] = values["Current_Value"] - values["Salary"]
    values["Stock_Change"] = values["Current_Value"] - values["Base_Value"]
    values["Value_Source"] = "ROS/Market"
    values.loc[mlb_match, "Value_Source"] = "MLB ROS Auction"
    values.loc[prospect_match, "Value_Source"] = "Prospect Posterior"
    values["Stock_Label"] = values["Stock_Label"].fillna("Standard")
    values["Role_Change"] = values["Role_Change"].fillna("Standard")
    values["Confidence_Label"] = values["Confidence_Label_MLB"].combine_first(values["Confidence_Label"]).fillna("Standard")

    status = load_mlb_status(mlb_status)
    if not status.empty and "MLBAMIDKey" in values.columns:
        values["MLBAMIDKey"] = values["MLBAMIDKey"].apply(normalized_player_id)
        status = status[status["MLBAMIDKey"].notna()].drop_duplicates("MLBAMIDKey")
        values = values.merge(status, on="MLBAMIDKey", how="left")
    for col, default in [
        ("MLB_Status", "Unknown"),
        ("Status_Flag", "UNKNOWN"),
        ("Current_Team", ""),
        ("Latest_Transaction_Date", ""),
        ("Latest_Transaction_Type", ""),
        ("Latest_Transaction_Description", ""),
    ]:
        if col not in values.columns:
            values[col] = default
        values[col] = values[col].fillna(default)
    return values, summary


def resolve_player(values, query, team=None):
    target = normalize_text(query)
    candidates = values.copy()
    if team:
        candidates = candidates[candidates["Team Name"].astype(str).str.lower().eq(str(team).lower())]
    exact = candidates[candidates["Name"].apply(normalize_text).eq(target)]
    if len(exact) == 1:
        return exact.iloc[0]
    contains = candidates[candidates["Name"].apply(normalize_text).str.contains(target, regex=False)]
    if len(contains) == 1:
        return contains.iloc[0]
    if contains.empty:
        raise ValueError(f"No player matched '{query}'.")
    names = contains[["Team Name", "Name"]].head(10).to_dict("records")
    raise ValueError(f"Ambiguous player '{query}'. Matches: {names}")


def print_values(values, limit=25, sort_by="Current_Surplus"):
    cols = [
        "Team Name", "Name", "Positions", "Salary", "Current_Value", "Current_Surplus",
        "Stock_Change", "Value_Source", "Confidence_Label",
    ]
    print(
        values.sort_values(sort_by, ascending=False)
        .head(limit)[cols]
        .to_string(index=False, float_format=lambda value: f"{value:.2f}")
    )


def main():
    parser = argparse.ArgumentParser(description="Unified player value table.")
    parser.add_argument("command", nargs="?", default="top", choices=["top", "export"])
    parser.add_argument("limit", nargs="?", type=int, default=25)
    parser.add_argument("--output", default="player_values.csv")
    parser.add_argument("--rosters", default=DEFAULT_ROSTERS)
    parser.add_argument("--hitters", default=DEFAULT_HITTERS)
    parser.add_argument("--pitchers", default=DEFAULT_PITCHERS)
    parser.add_argument("--avg", default=DEFAULT_AVG)
    parser.add_argument("--prospects", default=DEFAULT_PROSPECTS)
    parser.add_argument("--prospect-updates", default=DEFAULT_OUTPUT)
    parser.add_argument("--mlb-stock", default=MLB_STOCK_OUTPUT)
    parser.add_argument("--mlb-status", default=DEFAULT_MLB_STATUS)
    parser.add_argument("--data-source", choices=["static", "cache"], default=None)
    args = parser.parse_args()

    if args.data_source:
        paths = resolve_data_paths(args.data_source)
        args.rosters = paths.rosters
        args.hitters = paths.hitters_ros
        args.pitchers = paths.pitchers_ros
        args.avg = paths.avg_values
        args.prospects = paths.prospects
        args.mlb_stock = paths.mlb_stock
        args.mlb_status = paths.mlb_status

    values, _ = build_player_value_table(
        args.rosters, args.hitters, args.pitchers, args.avg, args.prospects,
        args.prospect_updates, args.mlb_stock, args.mlb_status
    )
    if args.command == "export":
        values.to_csv(args.output, index=False, float_format="%.2f")
        print(f"Wrote {args.output} with {len(values)} rows.")
        return
    print_values(values, limit=args.limit)


if __name__ == "__main__":
    main()
