import argparse
from datetime import date

import pandas as pd


HITTER_OUTPUT_COLUMNS = [
    "last_name, first_name", "player_id", "year", "pa", "xba", "xslg", "woba",
    "xwoba", "barrel_batted_rate", "hard_hit_percent", "whiff_percent",
]
PITCHER_OUTPUT_COLUMNS = HITTER_OUTPUT_COLUMNS + ["xera"]


def _require_pybaseball():
    try:
        from pybaseball import (
            statcast_batter_exitvelo_barrels,
            statcast_batter_expected_stats,
            statcast_pitcher_exitvelo_barrels,
            statcast_pitcher_expected_stats,
        )
    except ImportError as exc:
        raise RuntimeError("pybaseball is required to refresh Baseball Savant Statcast data.") from exc
    return {
        "batter_expected": statcast_batter_expected_stats,
        "pitcher_expected": statcast_pitcher_expected_stats,
        "batter_contact": statcast_batter_exitvelo_barrels,
        "pitcher_contact": statcast_pitcher_exitvelo_barrels,
    }


def _numeric(df, column):
    if column not in df.columns:
        return pd.NA
    return pd.to_numeric(df[column], errors="coerce")


def _empty(columns):
    return pd.DataFrame(columns=columns)


def normalize_expected_statcast(expected, contact, player_type):
    if expected is None or expected.empty:
        return _empty(PITCHER_OUTPUT_COLUMNS if player_type == "pitcher" else HITTER_OUTPUT_COLUMNS)

    out = expected.copy()
    if "last_name, first_name" not in out.columns and "player_name" in out.columns:
        out["last_name, first_name"] = out["player_name"]
    out["player_id"] = _numeric(out, "player_id").astype("Int64")
    out["pa"] = _numeric(out, "pa")
    out["xba"] = _numeric(out, "est_ba")
    out["xslg"] = _numeric(out, "est_slg")
    out["woba"] = _numeric(out, "woba")
    out["xwoba"] = _numeric(out, "est_woba")
    out["year"] = _numeric(out, "year").astype("Int64")

    if contact is not None and not contact.empty:
        contact_cols = contact.copy()
        contact_cols["player_id"] = _numeric(contact_cols, "player_id").astype("Int64")
        contact_cols["barrel_batted_rate"] = _numeric(contact_cols, "brl_percent")
        contact_cols["hard_hit_percent"] = _numeric(contact_cols, "ev95percent")
        out = out.merge(
            contact_cols[["player_id", "barrel_batted_rate", "hard_hit_percent"]],
            on="player_id",
            how="left",
        )
    else:
        out["barrel_batted_rate"] = pd.NA
        out["hard_hit_percent"] = pd.NA

    out["whiff_percent"] = pd.NA
    if player_type == "pitcher":
        out["xera"] = _numeric(out, "xera")
        columns = PITCHER_OUTPUT_COLUMNS
    else:
        columns = HITTER_OUTPUT_COLUMNS

    for col in columns:
        if col not in out.columns:
            out[col] = pd.NA
    return out[columns].sort_values("xwoba", ascending=player_type == "pitcher").reset_index(drop=True)


def fetch_statcast_leaderboards(year=None, hitter_min_pa=50, pitcher_min_pa=50, min_bbe=25, fetchers=None):
    year = int(year or date.today().year)
    fetchers = fetchers or _require_pybaseball()
    hitters = normalize_expected_statcast(
        fetchers["batter_expected"](year, hitter_min_pa),
        fetchers["batter_contact"](year, min_bbe),
        "hitter",
    )
    pitchers = normalize_expected_statcast(
        fetchers["pitcher_expected"](year, pitcher_min_pa),
        fetchers["pitcher_contact"](year, min_bbe),
        "pitcher",
    )
    return hitters, pitchers


def refresh_statcast_csvs(
    hitters_output,
    pitchers_output,
    year=None,
    hitter_min_pa=50,
    pitcher_min_pa=50,
    min_bbe=25,
    fetchers=None,
):
    hitters, pitchers = fetch_statcast_leaderboards(
        year=year,
        hitter_min_pa=hitter_min_pa,
        pitcher_min_pa=pitcher_min_pa,
        min_bbe=min_bbe,
        fetchers=fetchers,
    )
    hitters.to_csv(hitters_output, index=False, float_format="%.3f")
    pitchers.to_csv(pitchers_output, index=False, float_format="%.3f")
    return hitters, pitchers


def main():
    parser = argparse.ArgumentParser(description="Refresh YTD Baseball Savant expected Statcast CSVs.")
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--hitter-min-pa", type=int, default=50)
    parser.add_argument("--pitcher-min-pa", type=int, default=50)
    parser.add_argument("--min-bbe", type=int, default=25)
    parser.add_argument("--hitters-output", default="hitting_statcast_ytd_50_pa.csv")
    parser.add_argument("--pitchers-output", default="pitchers_statcast_ytd_30_ip.csv")
    args = parser.parse_args()

    hitters, pitchers = refresh_statcast_csvs(
        args.hitters_output,
        args.pitchers_output,
        year=args.year,
        hitter_min_pa=args.hitter_min_pa,
        pitcher_min_pa=args.pitcher_min_pa,
        min_bbe=args.min_bbe,
    )
    print(f"Wrote {args.hitters_output} with {len(hitters)} hitters.")
    print(f"Wrote {args.pitchers_output} with {len(pitchers)} pitchers.")


if __name__ == "__main__":
    main()
