import argparse
import re
import urllib.request
import warnings
from datetime import datetime, timezone

warnings.filterwarnings("ignore", message="Pandas requires version")
import pandas as pd


DEFAULT_PIPELINE_URL = "https://www.mlb.com/milb/prospects/{year}/top100/"


def fetch_html(url):
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 OttoneuIQ research script"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8", errors="replace")


def parse_visible_pipeline_rows(html, year, source_url):
    text = re.sub(r"<[^>]+>", "\n", html)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    try:
        start = lines.index("Rank")
    except ValueError:
        return pd.DataFrame()

    rows = []
    idx = start + 8
    while idx + 6 < len(lines):
        if not lines[idx].isdigit():
            idx += 1
            continue
        rank = int(lines[idx])
        if rank < 1 or rank > 100:
            idx += 1
            continue
        player = lines[idx + 1]
        position = lines[idx + 2]
        team = lines[idx + 3]
        level = lines[idx + 4]
        age = lines[idx + 5]
        bats = lines[idx + 6]
        throws = lines[idx + 7] if idx + 7 < len(lines) else ""
        if player == "Show Full List":
            break
        rows.append({
            "Rank": rank,
            "Name": player,
            "Position": position,
            "Org": team,
            "Level": level,
            "Age": age,
            "Bats": bats,
            "Throws": throws,
            "Source": f"MLB Pipeline Top 100 {year}",
            "SourceURL": source_url,
            "FetchedAtUTC": datetime.now(timezone.utc).isoformat(),
        })
        idx += 8
    return pd.DataFrame(rows)


def fetch_pipeline_top100(year=2025, url=None):
    source_url = url or DEFAULT_PIPELINE_URL.format(year=year)
    html = fetch_html(source_url)
    rows = parse_visible_pipeline_rows(html, year, source_url)
    if rows.empty:
        raise ValueError(f"No MLB Pipeline rows parsed from {source_url}")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Fetch visible MLB Pipeline Top 100 rows.")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--url", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    rows = fetch_pipeline_top100(args.year, args.url)
    output = args.output or f"mlb_pipeline_top100_{args.year}.csv"
    rows.to_csv(output, index=False)
    print(f"Wrote {output} with {len(rows)} rows.")
    if len(rows) < 100:
        print("Warning: MLB page only exposed visible rows. Full list may require a browser-rendered export or alternate endpoint.")


if __name__ == "__main__":
    main()
