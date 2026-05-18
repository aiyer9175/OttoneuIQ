import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore", message="Pandas requires version")
import pandas as pd

from data_sources import DEFAULT_CACHE_DIR, cache_path_for, refresh_data_cache, resolve_data_paths
from mlb_status import build_mlb_player_status, load_player_ids_from_csvs
from mlb_stock import build_mlb_stock
from player_trends import build_player_trend_table
from statcast_sources import refresh_statcast_csvs
from value_engine import build_player_value_table


PLAYER_VALUES_FILE = "player_values.csv"
PLAYER_TRENDS_FILE = "player_trends.csv"


def utc_timestamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _row_count(path):
    if not Path(path).exists():
        return 0
    return int(len(pd.read_csv(path)))


def _load_manifest(cache_dir):
    manifest_path = Path(cache_dir) / "manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text())


def _write_manifest(cache_dir, manifest):
    manifest_path = Path(cache_dir) / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))


def build_snapshot(
    source="static",
    cache_root=DEFAULT_CACHE_DIR,
    timestamp=None,
    validate=True,
    rebuild_stock=True,
    rebuild_status=None,
    rebuild_statcast=None,
):
    """Build one timestamped raw+derived data snapshot.

    The raw layer is delegated to data_sources.refresh_data_cache. Derived files
    are then rebuilt inside the same cache directory so downstream code can read
    a complete point-in-time view.
    """
    stamp = timestamp or utc_timestamp()
    cache_dir, manifest = refresh_data_cache(
        source=source,
        cache_root=cache_root,
        timestamp=stamp,
        validate=validate,
    )
    paths = resolve_data_paths("cache", cache_root=cache_root)

    statcast_error = None
    if rebuild_statcast is None:
        rebuild_statcast = source == "remote"
    if rebuild_statcast:
        try:
            refresh_statcast_csvs(
                paths.hitters_statcast,
                paths.pitchers_statcast,
            )
        except Exception as exc:
            statcast_error = str(exc)
            raise

    status_error = None
    if rebuild_status is None:
        rebuild_status = source == "remote"
    status_path = cache_path_for(cache_dir, "mlb_status")
    if rebuild_status:
        try:
            player_ids = load_player_ids_from_csvs([
                paths.hitters_ros,
                paths.pitchers_ros,
                paths.hitters_ytd,
                paths.pitchers_ytd,
                paths.relievers_ytd,
                paths.hitters_statcast,
                paths.pitchers_statcast,
            ])
            status = build_mlb_player_status(player_ids)
            status.to_csv(status_path, index=False)
        except Exception as exc:
            status_error = str(exc)

    if rebuild_stock:
        stock = build_mlb_stock(
            hitters_ros=paths.hitters_ros,
            pitchers_ros=paths.pitchers_ros,
            hitters_ytd=paths.hitters_ytd,
            hitters_statcast=paths.hitters_statcast,
            pitchers_ytd=paths.pitchers_ytd,
            pitchers_statcast=paths.pitchers_statcast,
            relievers_ytd=paths.relievers_ytd,
            hitters_ytd_value=paths.hitters_ytd_value,
            pitchers_ytd_value=paths.pitchers_ytd_value,
        )
        stock_path = cache_path_for(cache_dir, "mlb_stock")
        stock.to_csv(stock_path, index=False, float_format="%.3f")
    else:
        stock_path = paths.mlb_stock
        stock = pd.read_csv(stock_path)

    values, summary = build_player_value_table(
        rosters=paths.rosters,
        hitters=paths.hitters_ros,
        pitchers=paths.pitchers_ros,
        avg=paths.avg_values,
        prospects=paths.prospects,
        mlb_stock=str(stock_path),
        mlb_status=str(status_path),
    )
    trends = build_player_trend_table(values=values, stock=stock)

    values_path = Path(cache_dir) / PLAYER_VALUES_FILE
    trends_path = Path(cache_dir) / PLAYER_TRENDS_FILE
    summary_path = Path(cache_dir) / "team_summary.csv"
    values.to_csv(values_path, index=False, float_format="%.3f")
    trends.to_csv(trends_path, index=False, float_format="%.3f")
    summary.to_csv(summary_path, index=False, float_format="%.3f")

    manifest = _load_manifest(cache_dir) or manifest
    manifest["snapshot"] = {
        "created_at": stamp,
        "source": source,
        "derived_files": {
            "mlb_stock": {
                "file": "mlb_stock_values.csv",
                "rows": _row_count(stock_path),
                "rebuilt": bool(rebuild_stock),
            },
            "statcast": {
                "hitters_file": "hitting_statcast_ytd_50_pa.csv",
                "hitters_rows": _row_count(paths.hitters_statcast),
                "pitchers_file": "pitchers_statcast_ytd_30_ip.csv",
                "pitchers_rows": _row_count(paths.pitchers_statcast),
                "rebuilt": bool(rebuild_statcast and statcast_error is None),
                "error": statcast_error,
            },
            "mlb_status": {
                "file": "mlb_player_status.csv",
                "rows": _row_count(status_path),
                "rebuilt": bool(rebuild_status and status_error is None),
                "error": status_error,
            },
            "player_values": {
                "file": PLAYER_VALUES_FILE,
                "rows": _row_count(values_path),
            },
            "player_trends": {
                "file": PLAYER_TRENDS_FILE,
                "rows": _row_count(trends_path),
            },
            "team_summary": {
                "file": "team_summary.csv",
                "rows": _row_count(summary_path),
            },
        },
    }
    _write_manifest(cache_dir, manifest)
    return Path(cache_dir), manifest


def snapshot_status(cache_root=DEFAULT_CACHE_DIR):
    cache_root = Path(cache_root)
    if not cache_root.exists():
        return pd.DataFrame(columns=["Snapshot", "Player_Trends", "Player_Values", "Manifest"])
    rows = []
    for cache_dir in sorted(path for path in cache_root.iterdir() if path.is_dir()):
        rows.append({
            "Snapshot": cache_dir.name,
            "Player_Trends": (cache_dir / PLAYER_TRENDS_FILE).exists(),
            "Player_Values": (cache_dir / PLAYER_VALUES_FILE).exists(),
            "Manifest": (cache_dir / "manifest.json").exists(),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Build timestamped OttoneuIQ data snapshots.")
    parser.add_argument("command", choices=["build", "status"])
    parser.add_argument("--source", choices=["static", "remote"], default="static")
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--no-validate", action="store_true")
    parser.add_argument("--no-rebuild-stock", action="store_true")
    parser.add_argument("--rebuild-status", action="store_true", help="Fetch MLB active roster/transaction status.")
    parser.add_argument("--rebuild-statcast", action="store_true", help="Fetch current YTD Baseball Savant Statcast data.")
    parser.add_argument("--no-rebuild-statcast", action="store_true", help="Skip automatic Statcast refresh for remote snapshots.")
    args = parser.parse_args()

    if args.command == "status":
        print(snapshot_status(args.cache_root).to_string(index=False))
        return

    cache_dir, manifest = build_snapshot(
        source=args.source,
        cache_root=args.cache_root,
        timestamp=args.timestamp,
        validate=not args.no_validate,
        rebuild_stock=not args.no_rebuild_stock,
        rebuild_status=args.rebuild_status or None,
        rebuild_statcast=True if args.rebuild_statcast else False if args.no_rebuild_statcast else None,
    )
    print(f"Built snapshot: {cache_dir}")
    print(json.dumps(manifest["snapshot"], indent=2))


if __name__ == "__main__":
    main()
