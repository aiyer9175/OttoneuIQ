import argparse
import os
import warnings

warnings.filterwarnings("ignore", message="Pandas requires version")
import pandas as pd

from prospect_updates import normalize_text


DEFAULT_HITTERS_PRESEASON = "oopsy_auctioncalc_hitters_preseason.csv"
DEFAULT_HITTERS_ROS = "oopsy_auction_calc_hitters_ros.csv"
DEFAULT_PITCHERS_PRESEASON = "oopsy_auction_calc_pitchers_preseason.csv"
DEFAULT_PITCHERS_ROS = "oopsy_auction_calc_pitchers_all_ros.csv"
DEFAULT_HITTERS_YTD = "hitters_ytd_50_pa.csv"
DEFAULT_HITTERS_STATCAST = "hitting_statcast_ytd_50_pa.csv"
DEFAULT_PITCHERS_YTD = "pitchers_ytd_30_ip.csv"
DEFAULT_PITCHERS_STATCAST = "pitchers_statcast_ytd_30_ip.csv"
DEFAULT_RELIEVERS_YTD = "relievers_qualified.csv"
DEFAULT_HITTERS_YTD_VALUE = "ytd_fgpts_hitters.csv"
DEFAULT_PITCHERS_YTD_VALUE = "ytd_fgpts_pitchers.csv"
DEFAULT_OUTPUT = "mlb_stock_values.csv"


def normalized_player_id(value):
    if pd.isna(value):
        return None
    try:
        return str(int(float(value)))
    except ValueError:
        return str(value).strip() or None


def positions_set(value):
    return {part.strip().upper() for part in str(value or "").split("/") if part.strip()}


def read_optional_csv(path):
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def load_auction_values(path, player_type, prefix):
    df = pd.read_csv(path)
    df["PlayerIdKey"] = df["PlayerId"].apply(normalized_player_id)
    df["MLBAMIDKey"] = df["MLBAMID"].apply(normalized_player_id)
    df["NameKey"] = df["Name"].apply(normalize_text)
    df["Player_Type"] = player_type
    playing_time_col = "PA" if player_type == "hitter" else "IP"
    out = df[[
        "PlayerIdKey", "MLBAMIDKey", "NameKey", "Name", "Team", "POS",
        playing_time_col, "rPTS", "PTS", "Dollars", "Player_Type",
    ]].copy()
    out = out.sort_values("Dollars", ascending=False).drop_duplicates("PlayerIdKey", keep="first")
    return out.rename(columns={
        "Name": f"{prefix}_Name",
        "Team": f"{prefix}_Team",
        "POS": f"{prefix}_POS",
        playing_time_col: f"{prefix}_{playing_time_col}",
        "rPTS": f"{prefix}_rPTS",
        "PTS": f"{prefix}_PTS",
        "Dollars": f"{prefix}_Value",
    })


def load_ytd_auction_values(hitters_path=DEFAULT_HITTERS_YTD_VALUE, pitchers_path=DEFAULT_PITCHERS_YTD_VALUE):
    frames = []
    for path, player_type, playing_time_col in [
        (hitters_path, "hitter", "PA"),
        (pitchers_path, "pitcher", "IP"),
    ]:
        if not path or not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df["PlayerIdKey"] = df["PlayerId"].apply(normalized_player_id)
        df["MLBAMIDKey"] = df["MLBAMID"].apply(normalized_player_id)
        df["YTD_Value"] = pd.to_numeric(df["Dollars"], errors="coerce")
        df[f"YTD_{playing_time_col}_Auction"] = pd.to_numeric(df[playing_time_col], errors="coerce")
        frames.append(df[["PlayerIdKey", "MLBAMIDKey", "YTD_Value", f"YTD_{playing_time_col}_Auction"]])
    if not frames:
        return pd.DataFrame(columns=["PlayerIdKey", "MLBAMIDKey", "YTD_Value"])
    return pd.concat(frames, ignore_index=True, sort=False).drop_duplicates("PlayerIdKey", keep="first")


def load_hitters_ytd(path=DEFAULT_HITTERS_YTD, statcast_path=DEFAULT_HITTERS_STATCAST):
    ytd = read_optional_csv(path)
    if ytd.empty:
        return ytd
    ytd["PlayerIdKey"] = ytd["PlayerId"].apply(normalized_player_id)
    ytd["MLBAMIDKey"] = ytd["MLBAMID"].apply(normalized_player_id)
    ytd = ytd.rename(columns={
        "HR": "YTD_HR",
        "R": "YTD_R",
        "RBI": "YTD_RBI",
        "SB": "YTD_SB",
        "PA": "YTD_PA",
        "wRC+": "YTD_wRC+",
        "xwOBA": "YTD_xwOBA",
        "K%": "YTD_K%",
        "BB%": "YTD_BB%",
        "WAR": "YTD_WAR",
        "ISO": "YTD_ISO",
        "AVG": "YTD_AVG",
        "OBP": "YTD_OBP",
        "SLG": "YTD_SLG",
    })

    statcast = read_optional_csv(statcast_path)
    if not statcast.empty:
        statcast["MLBAMIDKey"] = statcast["player_id"].apply(normalized_player_id)
        statcast = statcast.rename(columns={
            "xwoba": "SC_xwOBA",
            "xslg": "SC_xSLG",
            "barrel_batted_rate": "SC_Barrel%",
            "hard_hit_percent": "SC_HardHit%",
            "whiff_percent": "SC_Whiff%",
            "avg_swing_speed": "SC_SwingSpeed",
        })
        keep = [
            "MLBAMIDKey", "SC_xwOBA", "SC_xSLG", "SC_Barrel%",
            "SC_HardHit%", "SC_Whiff%", "SC_SwingSpeed",
        ]
        for col in keep:
            if col not in statcast.columns:
                statcast[col] = pd.NA
        ytd = ytd.merge(statcast[keep], on="MLBAMIDKey", how="left")
    return ytd


def load_pitchers_ytd(path=DEFAULT_PITCHERS_YTD, statcast_path=DEFAULT_PITCHERS_STATCAST, relievers_path=DEFAULT_RELIEVERS_YTD):
    starters = read_optional_csv(path)
    relievers = read_optional_csv(relievers_path)
    frames = []
    if not starters.empty:
        starters["YTD_Role_Source"] = "starter_qualified"
        frames.append(starters)
    if not relievers.empty:
        relievers["YTD_Role_Source"] = "reliever_qualified"
        frames.append(relievers)
    if not frames:
        return pd.DataFrame()

    ytd = pd.concat(frames, ignore_index=True)
    ytd["PlayerIdKey"] = ytd["PlayerId"].apply(normalized_player_id)
    ytd["MLBAMIDKey"] = ytd["MLBAMID"].apply(normalized_player_id)
    ytd = ytd.sort_values(["PlayerIdKey", "IP"], ascending=[True, False]).drop_duplicates("PlayerIdKey")
    ytd = ytd.rename(columns={
        "G": "YTD_G",
        "IP": "YTD_IP",
        "GS": "YTD_GS",
        "SV": "YTD_SV",
        "K/9": "YTD_K9",
        "BB/9": "YTD_BB9",
        "ERA": "YTD_ERA",
        "xERA": "YTD_xERA",
        "FIP": "YTD_FIP",
        "xFIP": "YTD_xFIP",
        "WAR": "YTD_WAR",
        "vFA (pi)": "YTD_vFA",
    })

    statcast = read_optional_csv(statcast_path)
    if not statcast.empty:
        statcast["MLBAMIDKey"] = statcast["player_id"].apply(normalized_player_id)
        statcast = statcast.rename(columns={
            "xwoba": "SC_xwOBA_Allowed",
            "xba": "SC_xBA_Allowed",
            "xslg": "SC_xSLG_Allowed",
            "whiff_percent": "SC_Whiff%",
            "barrel_batted_rate": "SC_Barrel%_Allowed",
            "hard_hit_percent": "SC_HardHit%_Allowed",
        })
        keep = [
            "MLBAMIDKey", "SC_xwOBA_Allowed", "SC_xBA_Allowed", "SC_xSLG_Allowed",
            "SC_Whiff%", "SC_Barrel%_Allowed", "SC_HardHit%_Allowed",
        ]
        for col in keep:
            if col not in statcast.columns:
                statcast[col] = pd.NA
        ytd = ytd.merge(statcast[keep], on="MLBAMIDKey", how="left")
    return ytd


def clamp(value, low, high):
    if pd.isna(value):
        return 0.0
    return max(low, min(high, value))


def hitter_skill_score(row):
    components = []
    for col, low, high in [
        ("YTD_AVG", 0.220, 0.330),
        ("YTD_OBP", 0.280, 0.420),
        ("YTD_SLG", 0.350, 0.600),
    ]:
        if pd.notna(row.get(col)):
            components.append(clamp((float(row[col]) - low) / (high - low), 0, 1))
    for col, low, high in [("YTD_xwOBA", 0.280, 0.430), ("SC_xwOBA", 0.280, 0.430)]:
        if pd.notna(row.get(col)):
            components.append(clamp((float(row[col]) - low) / (high - low), 0, 1))
    if pd.notna(row.get("SC_Barrel%")):
        components.append(clamp(float(row["SC_Barrel%"]) / 18, 0, 1))
    if pd.notna(row.get("SC_HardHit%")):
        components.append(clamp((float(row["SC_HardHit%"]) - 30) / 25, 0, 1))
    if pd.notna(row.get("YTD_BB%")) and pd.notna(row.get("YTD_K%")):
        components.append(clamp((float(row["YTD_BB%"]) - float(row["YTD_K%"]) + 0.18) / 0.28, 0, 1))
    return sum(components) / len(components) if components else 0.5


def pitcher_skill_score(row):
    components = []
    if pd.notna(row.get("YTD_K9")) and pd.notna(row.get("YTD_BB9")):
        components.append(clamp(((float(row["YTD_K9"]) - float(row["YTD_BB9"])) - 3.5) / 8, 0, 1))
    for col, low, high in [("YTD_xERA", 5.0, 2.2), ("YTD_FIP", 5.0, 2.2), ("YTD_xFIP", 5.0, 2.2)]:
        if pd.notna(row.get(col)):
            components.append(clamp((float(row[col]) - low) / (high - low), 0, 1))
    if pd.notna(row.get("SC_xwOBA_Allowed")):
        components.append(clamp((float(row["SC_xwOBA_Allowed"]) - 0.390) / (0.250 - 0.390), 0, 1))
    if pd.notna(row.get("SC_Whiff%")):
        components.append(clamp((float(row["SC_Whiff%"]) - 18) / 18, 0, 1))
    if pd.notna(row.get("SC_HardHit%_Allowed")):
        components.append(clamp((float(row["SC_HardHit%_Allowed"]) - 50) / (25 - 50), 0, 1))
    return sum(components) / len(components) if components else 0.5


def confidence(row):
    if row["Player_Type"] == "hitter":
        return clamp(float(row.get("YTD_PA", 0) or 0) / 150, 0, 1)
    if "RP" in positions_set(row.get("ROS_POS")) and "SP" not in positions_set(row.get("ROS_POS")):
        return clamp(float(row.get("YTD_IP", 0) or 0) / 20, 0, 1)
    return clamp(float(row.get("YTD_IP", 0) or 0) / 40, 0, 1)


def role_change(row):
    pre = positions_set(row.get("Preseason_POS"))
    ros = positions_set(row.get("ROS_POS"))
    ytd_gs = float(row.get("YTD_GS", 0) or 0)
    ytd_sv = float(row.get("YTD_SV", 0) or 0)

    if "RP" in pre and "SP" not in pre and "SP" in ros:
        return "RP_TO_SP"
    if "SP" in pre and "SP" not in ros and "RP" in ros:
        return "SP_TO_RP"
    if "RP" in ros and "SP" not in ros and ytd_sv >= 5:
        return "RP_CLOSER"
    if "SP" in ros and ytd_gs >= 2:
        return "STABLE_SP"
    if "RP" in ros and "SP" not in ros:
        return "STABLE_RP"
    return "STABLE"


def movement_clip(row):
    if pd.isna(row.get("Projection_Change")):
        return 0.0
    change = float(row["Projection_Change"])
    pre = float(row["Preseason_Value"])
    ros = float(row["ROS_Value"])
    role = row["Role_Change"]
    if pre < 0 and ros < 0:
        return clamp(change, -8, 8)
    if pre < -20 and role != "RP_TO_SP":
        return clamp(change, -12, 12)
    if role == "RP_TO_SP":
        return clamp(change, -20, 35)
    if role == "SP_TO_RP":
        return clamp(change, -35, 15)
    return clamp(change, -30, 30)


def stock_label(row):
    if pd.isna(row.get("Projection_Change")):
        return "Missing Anchor"
    change = float(row["MLB_Stock_Change"])
    role = row["Role_Change"]
    skill = float(row["Skill_Score"])
    if role == "RP_TO_SP" and change > 5:
        return "Role-Driven Riser"
    if role == "SP_TO_RP" and change < -5:
        return "Role Loss"
    if change >= 8 and skill >= 0.58:
        return "Skills-Backed Riser"
    if float(row.get("YTD_ROS_Gap", 0) or 0) >= 15 and skill >= 0.58:
        return "YTD Breakout, Projection Skeptical"
    if float(row.get("YTD_ROS_Gap", 0) or 0) >= 15:
        return "Banked Production, Projection Skeptical"
    if change >= 8:
        return "Projection Riser"
    if change <= -8 and skill <= 0.42:
        return "Skills-Backed Faller"
    if change <= -8:
        return "Projection Faller"
    if role == "RP_CLOSER":
        return "Closer Value"
    return "Stable"


def confidence_label(value):
    if value >= 0.75:
        return "High"
    if value >= 0.35:
        return "Medium"
    return "Low"


def build_mlb_stock(
    hitters_preseason=DEFAULT_HITTERS_PRESEASON,
    hitters_ros=DEFAULT_HITTERS_ROS,
    pitchers_preseason=DEFAULT_PITCHERS_PRESEASON,
    pitchers_ros=DEFAULT_PITCHERS_ROS,
    hitters_ytd=DEFAULT_HITTERS_YTD,
    hitters_statcast=DEFAULT_HITTERS_STATCAST,
    pitchers_ytd=DEFAULT_PITCHERS_YTD,
    pitchers_statcast=DEFAULT_PITCHERS_STATCAST,
    relievers_ytd=DEFAULT_RELIEVERS_YTD,
    hitters_ytd_value=DEFAULT_HITTERS_YTD_VALUE,
    pitchers_ytd_value=DEFAULT_PITCHERS_YTD_VALUE,
):
    hitter_pre = load_auction_values(hitters_preseason, "hitter", "Preseason")
    hitter_ros = load_auction_values(hitters_ros, "hitter", "ROS")
    pitcher_pre = load_auction_values(pitchers_preseason, "pitcher", "Preseason")
    pitcher_ros = load_auction_values(pitchers_ros, "pitcher", "ROS")

    preseason = pd.concat([hitter_pre, pitcher_pre], ignore_index=True)
    ros = pd.concat([hitter_ros, pitcher_ros], ignore_index=True)
    preseason = preseason.drop_duplicates("PlayerIdKey", keep="first")
    ros = ros.drop_duplicates("PlayerIdKey", keep="first")
    merged = preseason.merge(
        ros,
        on="PlayerIdKey",
        how="outer",
        suffixes=("", "_ROSKey"),
    )
    merged["MLBAMIDKey"] = merged["MLBAMIDKey"].combine_first(merged["MLBAMIDKey_ROSKey"])
    merged["NameKey"] = merged["NameKey"].combine_first(merged["NameKey_ROSKey"])
    merged["Player_Type"] = merged["Player_Type"].fillna(merged["Player_Type_ROSKey"])

    hitters_context = load_hitters_ytd(hitters_ytd, hitters_statcast)
    pitchers_context = load_pitchers_ytd(pitchers_ytd, pitchers_statcast, relievers_ytd)
    contexts = []
    if not hitters_context.empty:
        contexts.append(hitters_context)
    if not pitchers_context.empty:
        contexts.append(pitchers_context)
    if contexts:
        context = pd.concat(contexts, ignore_index=True, sort=False)
        merged = merged.merge(context, on=["PlayerIdKey", "MLBAMIDKey"], how="left", suffixes=("", "_YTD"))

    ytd_values = load_ytd_auction_values(hitters_ytd_value, pitchers_ytd_value)
    if not ytd_values.empty:
        merged = merged.merge(ytd_values.drop(columns=["MLBAMIDKey"]), on="PlayerIdKey", how="left")

    merged["Display_Name"] = merged["ROS_Name"].fillna(merged["Preseason_Name"])
    merged["Display_Team"] = merged["ROS_Team"].fillna(merged["Preseason_Team"])
    merged["Display_POS"] = merged["ROS_POS"].fillna(merged["Preseason_POS"])
    merged["Preseason_Value"] = pd.to_numeric(merged["Preseason_Value"], errors="coerce")
    merged["ROS_Value"] = pd.to_numeric(merged["ROS_Value"], errors="coerce")
    merged["Projection_Change"] = merged["ROS_Value"] - merged["Preseason_Value"]
    merged["YTD_Value"] = pd.to_numeric(merged["YTD_Value"], errors="coerce")
    merged["YTD_ROS_Gap"] = merged["YTD_Value"] - merged["ROS_Value"]
    merged["Skill_Score"] = merged.apply(
        lambda row: hitter_skill_score(row) if row["Player_Type"] == "hitter" else pitcher_skill_score(row),
        axis=1,
    )
    merged["Sample_Confidence"] = merged.apply(confidence, axis=1)
    merged["Skill_Adjustment"] = (merged["Skill_Score"] - 0.5) * 4.0 * merged["Sample_Confidence"]
    merged["Banked_Value_Signal"] = (
        merged["YTD_ROS_Gap"].clip(lower=0, upper=20).fillna(0)
        * merged["Sample_Confidence"]
        * (0.12 + (merged["Skill_Score"] >= 0.58).astype(float) * 0.08)
    )
    merged["Role_Change"] = merged.apply(role_change, axis=1)
    merged["Clipped_Projection_Change"] = merged.apply(movement_clip, axis=1)
    merged["MLB_Stock_Change"] = (
        merged["Clipped_Projection_Change"] + merged["Skill_Adjustment"] + merged["Banked_Value_Signal"]
    )
    missing_anchor = merged["Projection_Change"].isna()
    merged.loc[missing_anchor, "MLB_Stock_Change"] = 0.0
    merged.loc[missing_anchor, "Skill_Adjustment"] = 0.0
    merged.loc[missing_anchor, "Banked_Value_Signal"] = 0.0
    merged["Stock_Label"] = merged.apply(stock_label, axis=1)
    merged["Confidence_Label"] = merged["Sample_Confidence"].apply(confidence_label)

    columns = [
        "PlayerIdKey", "MLBAMIDKey", "NameKey", "Display_Name", "Display_Team", "Display_POS", "Player_Type",
        "Preseason_POS", "ROS_POS", "Preseason_PA", "ROS_PA", "Preseason_IP", "ROS_IP",
        "Preseason_Value", "ROS_Value", "Projection_Change", "Clipped_Projection_Change",
        "YTD_Value", "YTD_ROS_Gap", "Banked_Value_Signal",
        "Skill_Score", "Skill_Adjustment", "Sample_Confidence", "Confidence_Label",
        "Role_Change", "Stock_Label", "MLB_Stock_Change",
        "YTD_PA", "YTD_HR", "YTD_R", "YTD_RBI", "YTD_SB", "YTD_AVG", "YTD_OBP",
        "YTD_SLG", "YTD_G", "YTD_IP", "YTD_GS", "YTD_SV", "YTD_wRC+", "YTD_xwOBA",
        "YTD_xERA", "YTD_FIP", "YTD_xFIP", "SC_xwOBA", "SC_xwOBA_Allowed", "SC_Whiff%",
    ]
    for col in columns:
        if col not in merged.columns:
            merged[col] = pd.NA
    return merged[columns].sort_values("MLB_Stock_Change", ascending=False).reset_index(drop=True)


def print_board(df, mode="risers", limit=25):
    sort_by = "MLB_Stock_Change"
    ascending = mode == "fallers"
    if mode == "value":
        sort_by = "ROS_Value"
        ascending = False
    cols = [
        "Display_Name", "Display_Team", "Display_POS", "Preseason_Value", "ROS_Value",
        "YTD_Value", "YTD_ROS_Gap", "Projection_Change", "MLB_Stock_Change",
        "Role_Change", "Stock_Label", "Confidence_Label",
    ]
    print(
        df.sort_values(sort_by, ascending=ascending).head(limit)[cols]
        .to_string(index=False, float_format=lambda value: f"{value:.2f}")
    )


def main():
    parser = argparse.ArgumentParser(description="MLB preseason-to-ROS stock engine.")
    parser.add_argument("command", nargs="?", default="risers", choices=["risers", "fallers", "value", "export"])
    parser.add_argument("limit", nargs="?", type=int, default=25)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    stock = build_mlb_stock()
    if args.command == "export":
        stock.to_csv(args.output, index=False, float_format="%.3f")
        print(f"Wrote {args.output} with {len(stock)} rows.")
        return
    print_board(stock, mode=args.command, limit=args.limit)


if __name__ == "__main__":
    main()
