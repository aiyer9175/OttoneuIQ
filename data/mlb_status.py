import argparse
import json
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone

import pandas as pd


BASE_URL = "https://statsapi.mlb.com/api/v1"
DEFAULT_OUTPUT = "mlb_player_status.csv"
MLB_SPORT_ID = 1
ACTIVE_ROSTER_TYPES = ("active",)


STATUS_COLUMNS = [
    "MLBAMIDKey",
    "Player_Name",
    "MLB_Status",
    "Status_Flag",
    "Current_Team_ID",
    "Current_Team",
    "Current_Roster_Type",
    "Latest_Transaction_Date",
    "Latest_Transaction_Type",
    "Latest_Transaction_Description",
    "Status_Source",
    "FetchedAtUTC",
]


def normalized_player_id(value):
    if pd.isna(value):
        return None
    try:
        return str(int(float(value)))
    except ValueError:
        clean = str(value).strip()
        return clean or None


def _request_json(path, params=None, opener=None):
    opener = opener or urllib.request.urlopen
    url = f"{BASE_URL}{path}"
    if params:
        query = urllib.parse.urlencode({key: value for key, value in params.items() if value is not None})
        url = f"{url}?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": "OttoneuIQ data refresh"})
    with opener(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_mlb_teams(season=None, opener=None):
    payload = _request_json("/teams", {"sportId": MLB_SPORT_ID, "season": season}, opener=opener)
    return payload.get("teams", [])


def fetch_team_roster(team_id, roster_type="active", season=None, opener=None):
    payload = _request_json(
        f"/teams/{team_id}/roster",
        {"rosterType": roster_type, "season": season},
        opener=opener,
    )
    return payload.get("roster", [])


def fetch_transactions(start_date, end_date, sport_id=MLB_SPORT_ID, opener=None):
    payload = _request_json(
        "/transactions",
        {
            "sportId": sport_id,
            "startDate": start_date,
            "endDate": end_date,
        },
        opener=opener,
    )
    return payload.get("transactions", [])


def latest_transactions_by_player(transactions):
    latest = {}
    for tx in transactions:
        person = tx.get("person") or {}
        player_id = normalized_player_id(person.get("id"))
        if not player_id:
            continue
        existing = latest.get(player_id)
        tx_date = str(tx.get("date") or "")
        if existing is None or tx_date >= str(existing.get("date") or ""):
            latest[player_id] = tx
    return latest


def active_roster_index(teams, roster_type="active", season=None, opener=None):
    active = {}
    for team in teams:
        team_id = team.get("id")
        if not team_id:
            continue
        for entry in fetch_team_roster(team_id, roster_type=roster_type, season=season, opener=opener):
            person = entry.get("person") or {}
            player_id = normalized_player_id(person.get("id"))
            if not player_id:
                continue
            active[player_id] = {
                "Player_Name": person.get("fullName") or person.get("displayName") or "",
                "Current_Team_ID": team_id,
                "Current_Team": team.get("name") or team.get("clubName") or "",
                "Current_Roster_Type": roster_type,
            }
    return active


def classify_transaction(transaction):
    if not transaction:
        return "NOT_ACTIVE_UNKNOWN", "UNKNOWN"

    tx_type = str((transaction.get("typeDesc") or transaction.get("typeCode") or "")).lower()
    description = str(transaction.get("description") or "").lower()
    text = f"{tx_type} {description}"

    if any(term in text for term in ["optioned", "assigned to", "reassigned", "sent to minors", "minor league"]):
        return "MINORS", "SENT_DOWN"
    if any(term in text for term in ["recalled", "selected", "contract selected", "promoted"]):
        return "RECENT_MLB_MOVE", "RECENT_RECALL"
    if any(term in text for term in ["injured", "60-day", "15-day", "10-day", "il"]):
        return "INJURED_LIST", "IL"
    if "designated for assignment" in text or "dfa" in text:
        return "DFA", "DFA"
    if "released" in text:
        return "RELEASED", "RELEASED"
    return "NOT_ACTIVE_UNKNOWN", "UNKNOWN"


def status_rows_for_players(player_ids, active_index, transactions_by_player, fetched_at=None):
    fetched_at = fetched_at or datetime.now(timezone.utc).isoformat()
    rows = []
    for raw_player_id in sorted({normalized_player_id(value) for value in player_ids if normalized_player_id(value)}):
        active = active_index.get(raw_player_id)
        tx = transactions_by_player.get(raw_player_id)
        if active:
            mlb_status = "ACTIVE_MLB"
            status_flag = "ACTIVE"
        else:
            mlb_status, status_flag = classify_transaction(tx)

        person = (tx or {}).get("person") or {}
        rows.append({
            "MLBAMIDKey": raw_player_id,
            "Player_Name": (active or {}).get("Player_Name") or person.get("fullName") or "",
            "MLB_Status": mlb_status,
            "Status_Flag": status_flag,
            "Current_Team_ID": (active or {}).get("Current_Team_ID", ""),
            "Current_Team": (active or {}).get("Current_Team", ""),
            "Current_Roster_Type": (active or {}).get("Current_Roster_Type", ""),
            "Latest_Transaction_Date": (tx or {}).get("date", ""),
            "Latest_Transaction_Type": (tx or {}).get("typeDesc") or (tx or {}).get("typeCode") or "",
            "Latest_Transaction_Description": (tx or {}).get("description", ""),
            "Status_Source": "MLB Stats API",
            "FetchedAtUTC": fetched_at,
        })
    return pd.DataFrame(rows, columns=STATUS_COLUMNS)


def build_mlb_player_status(player_ids, season=None, lookback_days=45, end_date=None, opener=None):
    end = end_date or date.today()
    if isinstance(end, str):
        end = date.fromisoformat(end)
    start = end - timedelta(days=int(lookback_days))
    teams = fetch_mlb_teams(season=season or end.year, opener=opener)
    active = active_roster_index(teams, roster_type="active", season=season or end.year, opener=opener)
    transactions = fetch_transactions(start.isoformat(), end.isoformat(), opener=opener)
    return status_rows_for_players(player_ids, active, latest_transactions_by_player(transactions))


def load_player_ids_from_csvs(paths):
    ids = set()
    for path in paths:
        if not path:
            continue
        df = pd.read_csv(path)
        for column in ["MLBAMID", "MLBAMIDKey", "player_id"]:
            if column in df.columns:
                ids.update(df[column].apply(normalized_player_id).dropna().tolist())
    return ids


def main():
    parser = argparse.ArgumentParser(description="Build MLB active/minors status indicators from MLB Stats API.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--lookback-days", type=int, default=45)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("csvs", nargs="+", help="CSV files containing MLBAMID, MLBAMIDKey, or player_id columns.")
    args = parser.parse_args()

    player_ids = load_player_ids_from_csvs(args.csvs)
    status = build_mlb_player_status(
        player_ids,
        season=args.season,
        lookback_days=args.lookback_days,
        end_date=args.end_date,
    )
    status.to_csv(args.output, index=False)
    print(f"Wrote {args.output} with {len(status)} rows.")


if __name__ == "__main__":
    main()
