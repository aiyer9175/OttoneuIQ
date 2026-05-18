import argparse
import os
import warnings

warnings.filterwarnings("ignore", message="Pandas requires version")
import pandas as pd

from prospect_updates import normalize_org, normalize_text


DEFAULT_PS_FILES = [
    ("2025_ps_hitters.csv", 2025, "hitter"),
    ("2024_ps_hitters.csv", 2024, "hitter"),
    ("2025_ps_pitchers.csv", 2025, "pitcher"),
    ("2024_ps_pitchers.csv", 2024, "pitcher"),
]


def read_ps_file(path, year, player_type):
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    required = {"Name", "Org", "Age", "PS Score"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    df["NameKey"] = df["Name"].apply(normalize_text)
    df["OrgKey"] = df["Org"].apply(normalize_org)
    df["PS_Year"] = year
    df["PS_Player_Type"] = player_type
    df["PS_Score"] = pd.to_numeric(df["PS Score"], errors="coerce")
    df["PS_Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["PS_PA"] = pd.to_numeric(df.get("PA"), errors="coerce") if "PA" in df.columns else pd.NA
    df["PS_Pitches"] = pd.to_numeric(df.get("Pitches"), errors="coerce") if "Pitches" in df.columns else pd.NA
    keep = [
        "NameKey", "OrgKey", "PS_Year", "PS_Player_Type", "PS_Score", "PS_Age",
        "PS_PA", "PS_Pitches",
    ]
    for optional in ["Power", "Discipline", "Contact Quality", "Max Velo", "xwOBA", "wOBA"]:
        if optional in df.columns:
            df[f"PS_{optional.replace(' ', '_')}"] = pd.to_numeric(df[optional], errors="coerce")
            keep.append(f"PS_{optional.replace(' ', '_')}")
    return df[keep]


def load_ps_priors(files=DEFAULT_PS_FILES):
    frames = [read_ps_file(path, year, player_type) for path, year, player_type in files]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=[
            "NameKey", "OrgKey", "PS_Best_Score", "PS_Best_Year", "PS_Player_Type",
            "Prospect_Pedigree_Label", "Prospect_Pedigree_Notes",
        ])
    raw = pd.concat(frames, ignore_index=True, sort=False)
    raw = raw[raw["PS_Score"].notna()]
    if raw.empty:
        return pd.DataFrame()

    records = []
    for (name_key, org_key), group in raw.groupby(["NameKey", "OrgKey"], dropna=False):
        best = group.sort_values(["PS_Score", "PS_Year"], ascending=[False, False]).iloc[0]
        recent = group.sort_values("PS_Year", ascending=False).iloc[0]
        best_score = float(best["PS_Score"])
        player_type = best["PS_Player_Type"]
        if best_score >= 0.97:
            label = "Former Blue-Chip Prospect"
        elif best_score >= 0.90:
            label = "Strong Prospect Track Record"
        elif best_score >= 0.80:
            label = "Notable Prospect Track Record"
        else:
            label = ""
        records.append({
            "NameKey": name_key,
            "OrgKey": org_key,
            "PS_Best_Score": best_score,
            "PS_Best_Year": int(best["PS_Year"]),
            "PS_Recent_Score": float(recent["PS_Score"]),
            "PS_Recent_Year": int(recent["PS_Year"]),
            "PS_Player_Type": player_type,
            "PS_Best_Age": best.get("PS_Age"),
            "Prospect_Pedigree_Label": label,
            "Prospect_Pedigree_Notes": f"{label}, best {int(best['PS_Year'])} PS {best_score:.3f}" if label else "",
        })
    return pd.DataFrame(records)


def add_ps_priors(trends, priors=None):
    if priors is None:
        priors = load_ps_priors()
    if priors.empty:
        trends = trends.copy()
        trends["PS_Best_Score"] = pd.NA
        trends["Prospect_Pedigree_Label"] = ""
        trends["Prospect_Pedigree_Notes"] = ""
        return trends
    out = trends.copy()
    if "NameKey" not in out.columns:
        out["NameKey"] = out["Name"].apply(normalize_text)
    org_col = "MLB Team" if "MLB Team" in out.columns else "Team"
    out["OrgKey"] = out[org_col].apply(normalize_org) if org_col in out.columns else ""
    out = out.merge(priors, on=["NameKey", "OrgKey"], how="left")
    out["Prospect_Pedigree_Label"] = out["Prospect_Pedigree_Label"].fillna("")
    out["Prospect_Pedigree_Notes"] = out["Prospect_Pedigree_Notes"].fillna("")
    return out


def main():
    parser = argparse.ArgumentParser(description="Build Prospect Savant priors for young MLB players.")
    parser.add_argument("--output", default="young_player_priors.csv")
    args = parser.parse_args()

    priors = load_ps_priors()
    priors.to_csv(args.output, index=False, float_format="%.3f")
    print(f"Wrote {args.output} with {len(priors)} rows.")


if __name__ == "__main__":
    main()
