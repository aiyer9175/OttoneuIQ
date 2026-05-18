import os
import sys


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
sys.path.insert(0, os.path.abspath(DATA_DIR))

from data_sources import DATASETS, refresh_data_cache, resolve_data_paths, source_status


def test_static_paths_resolve_existing_files():
    paths = resolve_data_paths("static")
    resolved = paths.as_dict()

    assert set(resolved) == set(DATASETS)
    assert os.path.exists(resolved["batters_auction"])
    assert os.path.exists(resolved["rosters"])


def test_static_refresh_creates_valid_cache(tmp_path):
    cache_dir, manifest = refresh_data_cache(source="static", cache_root=tmp_path)
    paths = resolve_data_paths("cache", cache_root=tmp_path)
    status = source_status(cache_root=tmp_path)

    assert cache_dir.exists()
    assert manifest["source"] == "static"
    assert os.path.exists(paths.batters_auction)
    assert os.path.exists(paths.hitters_ros)
    assert status["latest_cache"] == str(cache_dir)
