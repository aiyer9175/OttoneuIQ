import argparse
import os
import warnings

warnings.filterwarnings("ignore", message="Pandas requires version")
import pandas as pd

from data_sources import resolve_data_paths
from prospect_updates import normalize_text
from value_engine import build_player_value_table, load_or_build_mlb_stock, resolve_player
from young_player_priors import add_ps_priors


DEFAULT_TREND_PA = 150
DEFAULT_STARTER_IP = 40
DEFAULT_RELIEVER_IP = 20
DEFAULT_SP_RANKS = "Pitch Report 2026 - ranks5_15.csv"
POSITION_ORDER = ["C", "1B", "2B", "3B", "SS", "OF", "SP", "RP", "UTIL", "DH"]
RANKABLE_POSITIONS = ["C", "1B", "2B", "3B", "SS", "OF", "SP", "RP"]


def _numeric(row, column, default=0.0):
    value = row.get(column, default)
    if pd.isna(value):
        return default
    return float(value)


def _clamp(value, low, high):
    return max(low, min(high, value))


def normalized_id(value):
    if pd.isna(value):
        return None
    try:
        return str(int(float(value)))
    except ValueError:
        clean = str(value).strip()
        return clean or None


def load_sp_rank_overrides(path=DEFAULT_SP_RANKS):
    if not path or not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    if "Eno" not in df.columns or "MLBAM id" not in df.columns:
        return {}
    ranks = df[["MLBAM id", "Eno"]].copy()
    ranks["MLBAMIDKey"] = ranks["MLBAM id"].apply(normalized_id)
    ranks["Eno"] = pd.to_numeric(ranks["Eno"], errors="coerce")
    ranks = ranks[ranks["MLBAMIDKey"].notna() & ranks["Eno"].notna()]
    return {
        row["MLBAMIDKey"]: int(row["Eno"])
        for _, row in ranks.drop_duplicates("MLBAMIDKey").iterrows()
    }


def trend_sample_label(row):
    player_type = str(row.get("Player_Type", "")).lower()
    if player_type == "hitter":
        pa = _numeric(row, "YTD_PA")
        if pa >= DEFAULT_TREND_PA:
            return "stable sample"
        if pa >= DEFAULT_TREND_PA * 0.5:
            return "building sample"
        return "thin sample"

    ip = _numeric(row, "YTD_IP")
    role = str(row.get("Role_Change", ""))
    target = DEFAULT_RELIEVER_IP if "RP" in role and "SP" not in role else DEFAULT_STARTER_IP
    if ip >= target:
        return "stable sample"
    if ip >= target * 0.5:
        return "building sample"
    return "thin sample"


def trend_trade_adjustment(row):
    """Small in-season overlay for trade context.

    This intentionally does not replace ROS value. It rewards/penalizes current
    skill and banked production with sample-size and track-record dampening.
    """
    current_value = _numeric(row, "Current_Value")
    skill_score = _numeric(row, "Skill_Score", 0.5)
    confidence = _numeric(row, "Sample_Confidence")
    ytd_gap = _numeric(row, "YTD_ROS_Gap")

    skill_component = (skill_score - 0.5) * 7.0 * confidence
    if ytd_gap >= 0:
        production_rate = 0.18 if skill_score >= 0.62 else 0.10 if skill_score >= 0.50 else 0.06
        production_component = min(ytd_gap, 20.0) * production_rate * confidence
    else:
        production_component = max(ytd_gap, -20.0) * 0.07 * confidence

    raw_adjustment = skill_component + production_component

    if raw_adjustment < 0:
        if current_value >= 30:
            raw_adjustment *= 0.45
        elif current_value >= 20:
            raw_adjustment *= 0.65
    elif current_value < 8 and skill_score >= 0.62:
        raw_adjustment *= 1.25

    if ytd_gap >= 12 and skill_score >= 0.62 and confidence >= 0.75:
        cap = max(4.0, min(8.0, max(abs(current_value) * 0.75, ytd_gap * 0.32)))
    else:
        cap = max(2.5, min(6.0, abs(current_value) * 0.14))
    return round(_clamp(raw_adjustment, -cap, cap), 2)


def trend_label(row):
    adjustment = _numeric(row, "Trend_Trade_Adjustment")
    skill = _numeric(row, "Skill_Score", 0.5)
    ytd_gap = _numeric(row, "YTD_ROS_Gap")
    current_value = _numeric(row, "Current_Value")
    confidence = _numeric(row, "Sample_Confidence")
    pedigree = str(row.get("Prospect_Pedigree_Label", "") or "")
    player_type = str(row.get("Player_Type", "")).lower()

    if confidence < 0.25:
        return "Watchlist, Thin Sample"
    if adjustment >= 2 and skill >= 0.60 and pedigree and player_type == "hitter":
        return "Pedigree-Backed Breakout"
    if adjustment >= 2 and skill >= 0.60:
        return "Skills-Backed Riser"
    if adjustment >= 1:
        return "Production Riser"
    if adjustment <= -2 and skill <= 0.42:
        return "Skills-Backed Warning"
    if adjustment <= -1 and current_value >= 20:
        return "Track Record Hold"
    if adjustment <= -1:
        return "Production Warning"
    if ytd_gap >= 10 and skill < 0.52:
        return "Banked Stats, Skills Skeptical"
    return "Stable Context"


def trend_notes(row):
    notes = []
    current_value = _numeric(row, "Current_Value")
    skill = _numeric(row, "Skill_Score", 0.5)
    ytd_gap = _numeric(row, "YTD_ROS_Gap")
    projection_change = _numeric(row, "Projection_Change")
    pedigree_notes = str(row.get("Prospect_Pedigree_Notes", "") or "")

    if current_value >= 25:
        notes.append("strong ROS anchor")
    if projection_change >= 5:
        notes.append("projection already moved up")
    elif projection_change <= -5:
        notes.append("projection already moved down")
    if skill >= 0.62:
        notes.append("skills support production")
    elif skill <= 0.40:
        notes.append("skills add risk")
    if ytd_gap >= 10:
        notes.append("banked YTD value")
    elif ytd_gap <= -10:
        notes.append("YTD lagging ROS")
    if pedigree_notes:
        notes.append(pedigree_notes)
    if not notes:
        notes.append("limited trend signal")
    return ", ".join(notes)


def position_list(value):
    positions = [part.strip().upper() for part in str(value or "").split("/") if part.strip()]
    return positions or ["UTIL"]


def clean_positions(value):
    return "/".join(position_list(value))


def primary_position(value):
    positions = position_list(value)
    for position in POSITION_ORDER:
        if position in positions:
            return position
    return positions[0]


def player_role(row):
    positions = set(position_list(row.get("Positions")))
    player_type = str(row.get("Player_Type", "")).lower()
    current_value = _numeric(row, "Current_Value")
    ytd_pa = _numeric(row, "YTD_PA")
    ytd_g = _numeric(row, "YTD_G")
    ytd_ip = _numeric(row, "YTD_IP")
    ytd_gs = _numeric(row, "YTD_GS")
    ytd_sv = _numeric(row, "YTD_SV")
    skill_score = _numeric(row, "Skill_Score", 0.5)
    start_rate = ytd_gs / ytd_g if ytd_g > 0 else 0.0
    ip_per_start = ytd_ip / ytd_gs if ytd_gs > 0 else 0.0

    if player_type == "pitcher" or positions.intersection({"SP", "RP"}):
        if ytd_sv >= 5:
            return "Closer"
        if ytd_g > 0 and ytd_gs == 0 and "RP" in positions:
            if current_value >= 8 or skill_score >= 0.70:
                return "High-Leverage RP"
            return "Middle Relief"
        if "SP" in positions and ytd_gs >= 5 and (start_rate >= 0.5 or ip_per_start >= 3.5):
            if current_value >= 25 and skill_score >= 0.62:
                return "Ace / SP1"
            if current_value >= 15 and skill_score >= 0.55:
                return "Impact Starter"
            if current_value >= 5:
                return "Rotation Starter"
            return "Back-End Starter"
        if "SP" in positions and ytd_gs >= 2 and ytd_ip >= 15:
            return "Swingman / Bulk"
        if "SP" in positions and "RP" not in positions and current_value >= 10:
            return "Projected Starter"
        if "RP" in positions and current_value >= 8:
            return "High-Leverage RP"
        if "RP" in positions:
            return "Middle Relief"
        if "SP" in positions and current_value >= 3:
            return "SP Depth"
        return "Pitching Depth"

    if current_value >= 25:
        return "Core Bat"
    if ytd_pa >= 150 and current_value >= 8:
        return "Everyday Bat"
    if ytd_pa >= 90 and current_value >= 4:
        return "Regular Role"
    if current_value >= 8:
        return "Projected Bat"
    if ytd_pa >= 90:
        return "Playing-Time Watch"
    return "Depth Bat"


def add_position_ranks(trends, sp_rank_overrides=None):
    sp_rank_overrides = sp_rank_overrides if sp_rank_overrides is not None else load_sp_rank_overrides()
    ranked = trends.copy()
    ranked["Primary_Position"] = ranked["Positions"].apply(primary_position)
    ranked["Player_Role"] = ranked.apply(player_role, axis=1)
    ranked["Position_Rank"] = (
        ranked.groupby("Primary_Position")["Context_Value"]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )
    if sp_rank_overrides:
        sp_mask = ranked["Primary_Position"].eq("SP") & ranked.get("MLBAMIDKey", pd.Series(index=ranked.index, dtype=object)).apply(normalized_id).isin(sp_rank_overrides)
        ranked.loc[sp_mask, "Position_Rank"] = (
            ranked.loc[sp_mask, "MLBAMIDKey"]
            .apply(lambda value: sp_rank_overrides.get(normalized_id(value)))
            .astype("Int64")
        )
    rank_maps = {}
    for position in RANKABLE_POSITIONS:
        mask = ranked["Positions"].apply(lambda value: position in position_list(value))
        position_ranks = (
            ranked.loc[mask, "Context_Value"]
            .rank(method="min", ascending=False)
            .astype("Int64")
        )
        rank_maps[position] = position_ranks.to_dict()

    def eligible_rank_display(row):
        parts = []
        for position in position_list(row.get("Positions")):
            if position not in rank_maps:
                continue
            if position == "SP":
                rank = sp_rank_overrides.get(normalized_id(row.get("MLBAMIDKey"))) if sp_rank_overrides else None
                if rank is None:
                    rank = rank_maps[position].get(row.name)
            else:
                rank = rank_maps[position].get(row.name)
            if pd.notna(rank):
                parts.append(f"{position} #{int(rank)}")
        return ", ".join(parts) if parts else pd.NA

    ranked["Eligible_Position_Ranks"] = ranked.apply(eligible_rank_display, axis=1)
    return ranked


def build_player_trend_table(values=None, stock=None, sp_ranks=DEFAULT_SP_RANKS):
    if values is None:
        values, _ = build_player_value_table()
    if stock is None:
        stock = load_or_build_mlb_stock()

    stock = stock.copy()
    if "PlayerIdKey" in stock.columns:
        stock = stock[stock["PlayerIdKey"].notna()].drop_duplicates("PlayerIdKey")

    trend_cols = [
        "PlayerIdKey", "MLBAMIDKey", "Player_Type", "Preseason_Value", "ROS_Value", "Projection_Change",
        "Clipped_Projection_Change", "YTD_Value", "YTD_ROS_Gap", "Banked_Value_Signal",
        "Skill_Score", "Skill_Adjustment", "Sample_Confidence", "Role_Change",
        "Stock_Label", "MLB_Stock_Change", "YTD_PA", "YTD_G", "YTD_IP", "YTD_GS", "YTD_SV",
        "YTD_HR", "YTD_R", "YTD_RBI", "YTD_SB", "YTD_AVG", "YTD_OBP", "YTD_SLG",
        "YTD_wRC+", "YTD_xwOBA", "YTD_xERA", "YTD_FIP", "YTD_xFIP", "SC_xwOBA",
        "SC_xwOBA_Allowed", "SC_Whiff%",
    ]
    for col in trend_cols:
        if col not in stock.columns:
            stock[col] = pd.NA

    overlap = [col for col in trend_cols if col in values.columns and col != "PlayerIdKey"]
    base = values.drop(columns=overlap, errors="ignore").copy()
    if "Positions" in base.columns:
        base["Positions"] = base["Positions"].apply(clean_positions)
    trends = base.merge(stock[trend_cols], on="PlayerIdKey", how="left", suffixes=("", "_Trend"))
    trends = add_ps_priors(trends)
    trends["Trend_Trade_Adjustment"] = trends.apply(trend_trade_adjustment, axis=1)
    trends["Context_Value"] = trends["Current_Value"] + trends["Trend_Trade_Adjustment"]
    trends["Trend_Label"] = trends.apply(trend_label, axis=1)
    trends["Trend_Sample"] = trends.apply(trend_sample_label, axis=1)
    trends["Trend_Notes"] = trends.apply(trend_notes, axis=1)
    return add_position_ranks(trends, sp_rank_overrides=load_sp_rank_overrides(sp_ranks))


def trend_waterfall_rows(player_row):
    current_value = _numeric(player_row, "Current_Value")
    adjustment = _numeric(player_row, "Trend_Trade_Adjustment")
    skill = (_numeric(player_row, "Skill_Score", 0.5) - 0.5) * 7.0 * _numeric(player_row, "Sample_Confidence")
    ytd_gap = _numeric(player_row, "YTD_ROS_Gap")
    confidence = _numeric(player_row, "Sample_Confidence")
    if ytd_gap >= 0:
        production = min(ytd_gap, 20.0) * 0.10 * confidence
    else:
        production = max(ytd_gap, -20.0) * 0.07 * confidence
    dampener = adjustment - skill - production
    return pd.DataFrame([
        {"Component": "ROS Value", "Dollars": round(current_value, 2)},
        {"Component": "Skill Signal", "Dollars": round(skill, 2)},
        {"Component": "YTD Signal", "Dollars": round(production, 2)},
        {"Component": "Dampener/Cap", "Dollars": round(dampener, 2)},
        {"Component": "Context Value", "Dollars": round(current_value + adjustment, 2)},
    ])


def resolve_trend_player(trends, query, team=None):
    if team:
        return resolve_player(trends, query, team=team)
    target = normalize_text(query)
    exact = trends[trends["Name"].apply(normalize_text).eq(target)]
    if len(exact) == 1:
        return exact.iloc[0]
    contains = trends[trends["Name"].apply(normalize_text).str.contains(target, regex=False)]
    if len(contains) == 1:
        return contains.iloc[0]
    if contains.empty:
        raise ValueError(f"No player matched '{query}'.")
    names = contains[["Team Name", "Name"]].head(10).to_dict("records")
    raise ValueError(f"Ambiguous player '{query}'. Matches: {names}")


def main():
    parser = argparse.ArgumentParser(description="Player trend context for trade values.")
    parser.add_argument("query", nargs="?", default=None)
    parser.add_argument("--team", default=None)
    parser.add_argument("--data-source", choices=["static", "cache"], default=None)
    args = parser.parse_args()

    paths = resolve_data_paths(args.data_source) if args.data_source else None
    if paths:
        values, _ = build_player_value_table(
            paths.rosters,
            paths.hitters_ros,
            paths.pitchers_ros,
            paths.avg_values,
            paths.prospects,
            mlb_stock=paths.mlb_stock,
        )
        stock = load_or_build_mlb_stock(paths.mlb_stock)
    else:
        values, _ = build_player_value_table()
        stock = load_or_build_mlb_stock()

    trends = build_player_trend_table(values, stock)
    if args.query:
        row = resolve_trend_player(trends, args.query, team=args.team)
        cols = [
            "Team Name", "Name", "Positions", "Salary", "Current_Value", "Context_Value",
            "Trend_Trade_Adjustment", "Trend_Label", "Trend_Sample", "YTD_Value",
            "YTD_ROS_Gap", "Skill_Score", "Role_Change", "Trend_Notes",
        ]
        print(row[[col for col in cols if col in row.index]].to_frame().T.to_string(index=False))
        return

    cols = [
        "Team Name", "Name", "Positions", "Salary", "Current_Value", "Context_Value",
        "Trend_Trade_Adjustment", "Trend_Label", "Trend_Sample", "Trend_Notes",
    ]
    print(
        trends.sort_values("Trend_Trade_Adjustment", ascending=False)
        .head(25)[cols]
        .to_string(index=False, float_format=lambda value: f"{value:.2f}")
    )


if __name__ == "__main__":
    main()
