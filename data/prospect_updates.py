import argparse
import glob
import os
import re
import unicodedata

import pandas as pd

from post_auction import load_average_values, money_to_float
from valuation import clean_prospect_name


DEFAULT_PIPELINE_FILE = "mlb_pipeline_top100_2026_05_13.csv"
DEFAULT_COMPOSITE_FILE = os.path.join("data", "Baseball Composite Prospect List 2026 - List.csv")
DEFAULT_AVG_FILE = "fgpts_avgvalues.csv"
DEFAULT_OUTPUT = "prospect_value_updates.csv"
DEFAULT_STUFF_PLUS_FILE = "stuff_plus.csv"

ORG_TO_ABBR = {
    "Arizona Diamondbacks": "ARI", "Athletics": "ATH", "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS", "Chicago Cubs": "CHC",
    "Chicago White Sox": "CHW", "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL", "Detroit Tigers": "DET", "Houston Astros": "HOU",
    "Kansas City Royals": "KCR", "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN",
    "New York Mets": "NYM", "New York Yankees": "NYY", "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT", "San Diego Padres": "SDP", "San Francisco Giants": "SFG",
    "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL", "Tampa Bay Rays": "TBR",
    "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR", "Washington Nationals": "WSN",
}


def normalize_text(value):
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def normalize_org(value):
    text = str(value or "").strip()
    return ORG_TO_ABBR.get(text, text).upper()


def normalize_pitcher_name(value):
    text = str(value or "").strip()
    if "," not in text:
        return text
    last, first = [part.strip() for part in text.split(",", 1)]
    return f"{first} {last}".strip()


def latest_file(pattern):
    matches = glob.glob(pattern)
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def parse_level_and_type(path):
    basename = os.path.basename(path).lower()
    level = "AAA" if basename.startswith("aaa_") else "A"
    player_type = "pitcher" if "pitcher" in basename else "hitter"
    return level, player_type


def load_pipeline_prior(path=DEFAULT_PIPELINE_FILE):
    df = pd.read_csv(path)
    df["NameKey"] = df["Name"].apply(normalize_text)
    df["OrgKey"] = df["Org"].apply(normalize_org)
    df["Pipeline_Rank"] = pd.to_numeric(df["Rank"], errors="coerce")
    df["MLBAMID"] = pd.to_numeric(df["MLBAMID"], errors="coerce").astype("Int64")
    return df[["Name", "NameKey", "OrgKey", "MLBAMID", "Pipeline_Rank", "Position", "Level", "ETA", "Age"]]


def load_composite_prior(path=DEFAULT_COMPOSITE_FILE):
    df = pd.read_csv(path)
    df["Name"] = df["Name"].apply(clean_prospect_name)
    df["NameKey"] = df["Name"].apply(normalize_text)
    df["OrgKey"] = df["Team"].apply(normalize_org)
    df["Composite_Rank"] = pd.to_numeric(df["Rank"], errors="coerce")
    df["Composite_Level"] = df["Highest Level"].fillna("").astype(str).str.upper()
    df["Composite_MLBAMID"] = pd.to_numeric(df["mlbam"], errors="coerce").astype("Int64")
    return df[["NameKey", "OrgKey", "Composite_Rank", "Composite_Level", "Composite_MLBAMID"]]


def load_market_prior(path=DEFAULT_AVG_FILE):
    avg = pd.read_csv(path)
    avg["NameKey"] = avg["Name"].apply(normalize_text)
    avg["OrgKey"] = avg["MLB Org"].apply(normalize_org)
    avg["Avg_Salary"] = avg["Avg Salary"].apply(money_to_float)
    avg["Last10_Salary"] = avg["Last 10"].apply(money_to_float)
    avg["Roster%"] = pd.to_numeric(avg["Roster%"], errors="coerce").fillna(0)
    return avg[["NameKey", "OrgKey", "Avg_Salary", "Last10_Salary", "Roster%"]]


def load_prospect_savant_files(paths=None):
    if paths is None:
        paths = [
            latest_file("AAA_hitters_YTD_*.csv"),
            latest_file("AAA_pitchers_YTD_*.csv"),
            latest_file("A_hitters_YTD_*.csv"),
            latest_file("A_pitchers_YTD_*.csv"),
        ]
    frames = []
    for path in [p for p in paths if p]:
        df = pd.read_csv(path)
        level, player_type = parse_level_and_type(path)
        df["Savant_File"] = os.path.basename(path)
        df["Savant_Level"] = level
        df["Player_Type"] = player_type
        df["NameKey"] = df["Name"].apply(normalize_text)
        df["OrgKey"] = df["Org"].apply(normalize_org)
        for source, target in [
            ("K%.1", "K_Rate"),
            ("BB%.1", "BB_Rate"),
            ("Whiff%.1", "Whiff_Rate"),
            ("SwStr%.1", "SwStr_Rate"),
        ]:
            if source in df.columns:
                df[target] = pd.to_numeric(df[source], errors="coerce")
        for source, target in [
            ("K%", "K_Rate"),
            ("BB%", "BB_Rate"),
            ("Whiff%", "Whiff_Rate"),
            ("SwStr%", "SwStr_Rate"),
        ]:
            if target not in df.columns and source in df.columns:
                df[target] = pd.to_numeric(df[source], errors="coerce")
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No Prospect Savant YTD files found.")
    return pd.concat(frames, ignore_index=True)


def weighted_average(group, column, weight_column="Stuff_Pitches"):
    values = pd.to_numeric(group[column], errors="coerce")
    weights = pd.to_numeric(group[weight_column], errors="coerce").fillna(0)
    valid = values.notna() & weights.gt(0)
    if valid.any():
        return (values[valid] * weights[valid]).sum() / weights[valid].sum()
    return values.mean()


def load_stuff_plus(path=DEFAULT_STUFF_PLUS_FILE):
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    required = {"player_name", "level", "n_pitches"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    df["Name"] = df["player_name"].apply(normalize_pitcher_name)
    df["NameKey"] = df["Name"].apply(normalize_text)
    df["Stuff_Level"] = df["level"].fillna("").astype(str).str.upper()
    df["Stuff_Pitches"] = pd.to_numeric(df["n_pitches"], errors="coerce").fillna(0)
    if "pitcher" in df.columns:
        df["Stuff_MLBAMID"] = pd.to_numeric(df["pitcher"], errors="coerce").astype("Int64")
    else:
        df["Stuff_MLBAMID"] = pd.Series([pd.NA] * len(df), dtype="Int64")

    rename = {
        "P+": "Pitching_Plus",
        "S+": "Stuff_Plus",
        "L+": "Command_Plus",
        "xRV100_P": "Pitching_xRV100",
        "xRV100_S": "Stuff_xRV100",
        "xRV100_L": "Command_xRV100",
        "xSwSt_P": "Pitching_xSwStr",
        "xWhiff_P": "Pitching_xWhiff",
    }
    for source, target in rename.items():
        if source in df.columns:
            df[target] = pd.to_numeric(df[source], errors="coerce")

    records = []
    metric_cols = [col for col in rename.values() if col in df.columns]
    for (name_key, level), group in df.groupby(["NameKey", "Stuff_Level"], dropna=False):
        record = {
            "NameKey": name_key,
            "Stuff_Level": level,
            "Stuff_Pitches": group["Stuff_Pitches"].sum(),
            "Stuff_MLBAMID": group["Stuff_MLBAMID"].dropna().iloc[0] if group["Stuff_MLBAMID"].notna().any() else pd.NA,
        }
        for col in metric_cols:
            record[col] = weighted_average(group, col)
        records.append(record)
    return pd.DataFrame(records)


def rank_value(rank, source):
    if pd.isna(rank):
        return 0.0
    rank = float(rank)
    if source == "pipeline":
        if rank <= 5:
            return 25.0
        if rank <= 20:
            return 15.0
        if rank <= 50:
            return 8.0
        if rank <= 100:
            return 3.0
        return 0.0
    if rank <= 5:
        return 22.0
    if rank <= 20:
        return 13.0
    if rank <= 50:
        return 7.0
    if rank <= 100:
        return 3.0
    if rank <= 300:
        return 1.5
    return 0.5


def prior_value(row):
    pipeline_value = rank_value(row.get("Pipeline_Rank"), "pipeline")
    composite_value = rank_value(row.get("Composite_Rank"), "composite")
    market_value = max(float(row.get("Avg_Salary", 0) or 0), float(row.get("Last10_Salary", 0) or 0) * 0.85)
    return max(pipeline_value, composite_value, market_value * 0.9)


def sample_confidence(row):
    if row["Player_Type"] == "hitter":
        sample = min(float(row.get("PA", 0) or 0) / 150.0, 1.0)
    else:
        sample = min(float(row.get("Pitches", 0) or 0) / 500.0, 1.0)
    level_weight = 1.0 if row["Savant_Level"] == "AAA" else 0.55
    age = float(row.get("Age", 0) or 0)
    expected_age = 25 if row["Savant_Level"] == "AAA" else 21
    age_weight = min(max(1.0 + (expected_age - age) * 0.06, 0.7), 1.25)
    confidence = min(sample * level_weight * age_weight, 1.0)
    if row["Player_Type"] == "pitcher" and pd.notna(row.get("Stuff_Pitches")):
        stuff_sample = min(float(row.get("Stuff_Pitches", 0) or 0) / 450.0, 1.0)
        confidence = max(confidence, min(stuff_sample * level_weight * age_weight, 1.0))
    return confidence


def bounded(value, lower=0.0, upper=1.0):
    return min(max(float(value), lower), upper)


def plus_signal(value, low=85.0, high=115.0):
    if pd.isna(value):
        return None
    return bounded((float(value) - low) / (high - low))


def rate_signal(value, low, high, lower_is_better=False):
    if pd.isna(value):
        return None
    number = float(value)
    if number <= 1.0:
        number *= 100.0
    if lower_is_better:
        return bounded((high - number) / (high - low))
    return bounded((number - low) / (high - low))


def average_present(values):
    present = [value for value in values if value is not None and not pd.isna(value)]
    if not present:
        return None
    return sum(present) / len(present)


def pitcher_pitch_quality(row):
    stuff = plus_signal(row.get("Stuff_Plus"))
    pitching = plus_signal(row.get("Pitching_Plus"))
    if stuff is not None and pitching is not None:
        return (stuff * 0.75) + (pitching * 0.25)
    return stuff if stuff is not None else pitching


def evidence_score(row):
    ps = float(row.get("PS Score", 0.5) or 0.5)
    xwoba = float(row.get("xwOBA", 0.5) or 0.5)
    if row["Player_Type"] == "pitcher":
        # Pitcher xwOBA is raw allowed wOBA, so lower is better.
        xwoba_signal = min(max((0.390 - xwoba) / 0.220, 0.0), 1.0)
        rate_quality = average_present([
            rate_signal(row.get("K_Rate"), 18.0, 35.0),
            rate_signal(row.get("BB_Rate"), 5.0, 15.0, lower_is_better=True),
            rate_signal(row.get("SwStr_Rate"), 8.0, 18.0),
            rate_signal(row.get("Whiff_Rate"), 20.0, 40.0),
        ])
        pitch_quality = pitcher_pitch_quality(row)
        if pitch_quality is None:
            pitch_quality = rate_quality
        if rate_quality is None:
            rate_quality = pitch_quality
        if pitch_quality is None:
            return (ps * 0.65) + (xwoba_signal * 0.35)

        evidence = (ps * 0.30) + (xwoba_signal * 0.20) + (rate_quality * 0.20) + (pitch_quality * 0.30)
        is_aaa_starter = row["Savant_Level"] == "AAA" and "SP" in str(row.get("Pos", ""))
        workload = max(float(row.get("Pitches", 0) or 0), float(row.get("Stuff_Pitches", 0) or 0))
        if is_aaa_starter and workload >= 450 and pitch_quality >= 0.60:
            evidence += 0.04
        return bounded(evidence)
    return (ps * 0.70) + (xwoba * 0.30)


def max_movement(row):
    prior = float(row["Prior_Value"])
    pitch_quality = pitcher_pitch_quality(row)
    is_aaa_starter = row["Player_Type"] == "pitcher" and row["Savant_Level"] == "AAA" and "SP" in str(row.get("Pos", ""))
    workload = max(float(row.get("Pitches", 0) or 0), float(row.get("Stuff_Pitches", 0) or 0))
    if is_aaa_starter and workload >= 450 and pitch_quality is not None and pitch_quality >= 0.60:
        return 9.0 if prior >= 15 else 7.0
    if prior >= 15:
        return 8.0
    if pd.notna(row.get("Pipeline_Rank")):
        return 5.0
    if pd.notna(row.get("Composite_Rank")):
        return 4.0
    return 4.0 if row["Savant_Level"] == "AAA" else 2.0


def build_prospect_updates(
    pipeline_csv=DEFAULT_PIPELINE_FILE,
    composite_csv=DEFAULT_COMPOSITE_FILE,
    avg_csv=DEFAULT_AVG_FILE,
    savant_files=None,
    stuff_plus_csv=DEFAULT_STUFF_PLUS_FILE,
):
    savant = load_prospect_savant_files(savant_files)
    pipeline = load_pipeline_prior(pipeline_csv) if pipeline_csv and os.path.exists(pipeline_csv) else pd.DataFrame()
    composite = load_composite_prior(composite_csv) if composite_csv and os.path.exists(composite_csv) else pd.DataFrame()
    market = load_market_prior(avg_csv) if avg_csv and os.path.exists(avg_csv) else pd.DataFrame()
    stuff = load_stuff_plus(stuff_plus_csv)

    merged = savant.merge(pipeline, on=["NameKey", "OrgKey"], how="left", suffixes=("", "_Pipeline"))
    merged = merged.merge(composite, on=["NameKey", "OrgKey"], how="left")
    merged = merged.merge(market, on=["NameKey", "OrgKey"], how="left")
    if not stuff.empty:
        merged = merged.merge(
            stuff,
            left_on=["NameKey", "Savant_Level"],
            right_on=["NameKey", "Stuff_Level"],
            how="left",
        )

    for col in ["Avg_Salary", "Last10_Salary", "Roster%"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0)
    merged["Prior_Value"] = merged.apply(prior_value, axis=1)
    merged["Evidence_Score"] = merged.apply(evidence_score, axis=1)
    merged["Confidence"] = merged.apply(sample_confidence, axis=1)
    merged["Max_Move"] = merged.apply(max_movement, axis=1)
    merged["Prospect_Update"] = (merged["Evidence_Score"] - 0.5) * 2.0 * merged["Max_Move"] * merged["Confidence"]
    merged["Updated_Prospect_Value"] = (merged["Prior_Value"] + merged["Prospect_Update"]).clip(lower=0)
    merged["Confidence_Label"] = pd.cut(
        merged["Confidence"],
        bins=[-0.01, 0.25, 0.55, 1.01],
        labels=["Low", "Medium", "High"],
    ).astype(str)
    merged["Match_Status"] = "name_org"
    merged.loc[merged["Pipeline_Rank"].isna() & merged["Composite_Rank"].isna(), "Match_Status"] = "savant_only"

    columns = [
        "Name", "Org", "Savant_Level", "Player_Type", "Pos", "Age", "PA", "Pitches",
        "PS Score", "xwOBA", "wOBA", "K_Rate", "BB_Rate", "Whiff_Rate", "SwStr_Rate",
        "Pitching_Plus", "Stuff_Plus", "Command_Plus", "Stuff_Pitches",
        "Evidence_Score", "Confidence", "Confidence_Label",
        "Pipeline_Rank", "Composite_Rank", "MLBAMID", "Composite_MLBAMID",
        "Prior_Value", "Prospect_Update", "Updated_Prospect_Value",
        "Avg_Salary", "Last10_Salary", "Roster%", "Match_Status", "Savant_File",
    ]
    for col in columns:
        if col not in merged.columns:
            merged[col] = pd.NA
    return merged[columns].sort_values(
        ["Updated_Prospect_Value", "Prospect_Update"], ascending=False
    ).reset_index(drop=True)


def print_table(df, limit=25):
    cols = [
        "Name", "Org", "Savant_Level", "Player_Type", "Pos", "Pipeline_Rank",
        "Prior_Value", "Prospect_Update", "Updated_Prospect_Value", "Confidence_Label",
    ]
    print(df.head(limit)[cols].to_string(index=False, float_format=lambda value: f"{value:.2f}"))


def main():
    parser = argparse.ArgumentParser(description="Prospect prior/posterior value updater.")
    parser.add_argument("command", nargs="?", default="top", choices=["top", "risers", "fallers", "export"])
    parser.add_argument("limit", nargs="?", type=int, default=25)
    parser.add_argument("--pipeline", default=DEFAULT_PIPELINE_FILE)
    parser.add_argument("--composite", default=DEFAULT_COMPOSITE_FILE)
    parser.add_argument("--avg", default=DEFAULT_AVG_FILE)
    parser.add_argument("--stuff-plus", default=DEFAULT_STUFF_PLUS_FILE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    updates = build_prospect_updates(args.pipeline, args.composite, args.avg, stuff_plus_csv=args.stuff_plus)
    if args.command == "export":
        updates.to_csv(args.output, index=False, float_format="%.3f")
        print(f"Wrote {args.output} with {len(updates)} rows.")
        return
    if args.command == "risers":
        updates = updates.sort_values("Prospect_Update", ascending=False)
    elif args.command == "fallers":
        updates = updates.sort_values("Prospect_Update", ascending=True)
    print_table(updates, limit=args.limit)


if __name__ == "__main__":
    main()
