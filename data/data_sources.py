import argparse
import json
import os
import shutil
import urllib.request
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore", message="Pandas requires version")

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
DEFAULT_CACHE_DIR = REPO_ROOT / "data_cache"
ENV_PREFIX = "OTTONEUIQ"

DATASETS = {
    "batters_auction": "batters_auctioncalc.csv",
    "pitchers_auction": "pitchers_auctioncalc.csv",
    "prospects": "Baseball Composite Prospect List 2026 - List.csv",
    "rosters": "current_rosters.csv",
    "hitters_ros": "hitters_ros.csv",
    "pitchers_ros": "pitchers_ros.csv",
    "hitters_ytd": "hitters_ytd_50_pa.csv",
    "pitchers_ytd": "pitchers_ytd_30_ip.csv",
    "relievers_ytd": "relievers_qualified.csv",
    "hitters_statcast": "hitting_statcast_ytd_50_pa.csv",
    "pitchers_statcast": "pitchers_statcast_ytd_30_ip.csv",
    "hitters_ytd_value": "ytd_fgpts_hitters.csv",
    "pitchers_ytd_value": "ytd_fgpts_pitchers.csv",
    "avg_values": "fgpts_avgvalues.csv",
    "pipeline": "mlb_pipeline_top100_2026_05_13.csv",
    "stuff_plus": "stuff_plus.csv",
    "mlb_stock": "mlb_stock_values.csv",
    "mlb_status": "mlb_player_status.csv",
}

ROOT_STATIC_DATASETS = {
    "rosters", "hitters_ros", "pitchers_ros", "hitters_ytd", "pitchers_ytd",
    "relievers_ytd", "hitters_statcast", "pitchers_statcast",
    "hitters_ytd_value", "pitchers_ytd_value", "avg_values", "pipeline",
    "stuff_plus", "mlb_stock", "mlb_status"
}

REQUIRED_COLUMNS = {
    "batters_auction": {"Name", "POS", "Dollars"},
    "pitchers_auction": {"Name", "POS", "Dollars"},
    "prospects": {"Name", "Team", "Pos", "Rank"},
    "rosters": {"Name", "Salary"},
    "hitters_ros": {"Name", "POS", "Dollars"},
    "pitchers_ros": {"Name", "POS", "Dollars"},
    "hitters_ytd": {"Name", "PlayerId", "MLBAMID"},
    "pitchers_ytd": {"Name", "PlayerId", "MLBAMID"},
    "relievers_ytd": {"PlayerId", "MLBAMID"},
    "hitters_statcast": {"player_id"},
    "pitchers_statcast": {"player_id"},
    "hitters_ytd_value": {"PlayerId", "MLBAMID", "Dollars"},
    "pitchers_ytd_value": {"PlayerId", "MLBAMID", "Dollars"},
    "avg_values": {"Name", "Avg Salary", "Last 10"},
    "stuff_plus": {"player_name", "level", "n_pitches"},
    "mlb_status": {"MLBAMIDKey", "MLB_Status", "Status_Flag"},
}


@dataclass(frozen=True)
class DataPaths:
    batters_auction: str
    pitchers_auction: str
    prospects: str
    rosters: str
    hitters_ros: str
    pitchers_ros: str
    hitters_ytd: str
    pitchers_ytd: str
    relievers_ytd: str
    hitters_statcast: str
    pitchers_statcast: str
    hitters_ytd_value: str
    pitchers_ytd_value: str
    avg_values: str
    pipeline: str
    stuff_plus: str
    mlb_stock: str
    mlb_status: str

    def as_dict(self):
        return self.__dict__.copy()


def static_path(dataset_key):
    filename = DATASETS[dataset_key]
    root_path = REPO_ROOT / filename
    data_path = DATA_DIR / filename
    if dataset_key in ROOT_STATIC_DATASETS and root_path.exists():
        return root_path
    return data_path


def remote_env_name(dataset_key):
    return f"{ENV_PREFIX}_{dataset_key.upper()}_URL"


def remote_url(dataset_key):
    return os.getenv(remote_env_name(dataset_key), "").strip()


def latest_cache_dir(cache_dir=DEFAULT_CACHE_DIR):
    cache_root = Path(cache_dir)
    if not cache_root.exists():
        return None
    candidates = [path for path in cache_root.iterdir() if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.name)


def cache_path_for(cache_dir, dataset_key):
    return Path(cache_dir) / DATASETS[dataset_key]


def validate_csv(path, dataset_key):
    required = REQUIRED_COLUMNS.get(dataset_key)
    if not required:
        return
    df = pd.read_csv(path, nrows=1)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns for {dataset_key}: {sorted(missing)}")


def copy_static_dataset(dataset_key, destination):
    src = static_path(dataset_key)
    if not src.exists():
        raise FileNotFoundError(f"Static dataset not found for {dataset_key}: {src}")
    shutil.copyfile(src, destination)


def download_dataset(dataset_key, destination):
    url = remote_url(dataset_key)
    if not url:
        copy_static_dataset(dataset_key, destination)
        return "static_fallback"
    with urllib.request.urlopen(url, timeout=60) as response:
        destination.write_bytes(response.read())
    return "remote"


def refresh_data_cache(source="remote", cache_root=DEFAULT_CACHE_DIR, timestamp=None, validate=True):
    stamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cache_dir = Path(cache_root) / stamp
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": stamp,
        "source": source,
        "datasets": {},
    }
    for dataset_key in DATASETS:
        destination = cache_path_for(cache_dir, dataset_key)
        if source == "static":
            used_source = "static"
            copy_static_dataset(dataset_key, destination)
        elif source == "remote":
            used_source = download_dataset(dataset_key, destination)
        else:
            raise ValueError("source must be 'static' or 'remote'")
        if validate:
            validate_csv(destination, dataset_key)
        manifest["datasets"][dataset_key] = {
            "file": DATASETS[dataset_key],
            "source": used_source,
            "url_env": remote_env_name(dataset_key),
            "url_configured": bool(remote_url(dataset_key)),
        }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return cache_dir, manifest


def resolve_data_paths(source="static", cache_root=DEFAULT_CACHE_DIR):
    if source == "static":
        paths = {key: str(static_path(key)) for key in DATASETS}
    elif source == "cache":
        cache_dir = latest_cache_dir(cache_root)
        if cache_dir is None:
            paths = {key: str(static_path(key)) for key in DATASETS}
        else:
            paths = {
                key: str(cache_path_for(cache_dir, key))
                if cache_path_for(cache_dir, key).exists()
                else str(static_path(key))
                for key in DATASETS
            }
    else:
        raise ValueError("source must be 'static' or 'cache'")
    return DataPaths(**paths)


def source_status(cache_root=DEFAULT_CACHE_DIR):
    cache_dir = latest_cache_dir(cache_root)
    if cache_dir is None:
        return {"latest_cache": None, "datasets": {}}
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    return {
        "latest_cache": str(cache_dir),
        "datasets": manifest.get("datasets", {}),
    }


def main():
    parser = argparse.ArgumentParser(description="Refresh OttoneuIQ data cache from static files or configured remote CSV URLs.")
    parser.add_argument("command", choices=["refresh", "status"])
    parser.add_argument("--source", choices=["static", "remote"], default="remote")
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()

    if args.command == "status":
        print(json.dumps(source_status(args.cache_root), indent=2))
        return

    cache_dir, manifest = refresh_data_cache(
        source=args.source,
        cache_root=args.cache_root,
        validate=not args.no_validate,
    )
    print(f"Refreshed data cache: {cache_dir}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
