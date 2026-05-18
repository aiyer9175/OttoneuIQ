import os
import sys


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
sys.path.insert(0, os.path.abspath(DATA_DIR))

from snapshots import PLAYER_TRENDS_FILE, PLAYER_VALUES_FILE, build_snapshot, snapshot_status


def test_build_snapshot_writes_derived_files(tmp_path):
    cache_dir, manifest = build_snapshot(
        source="static",
        cache_root=tmp_path,
        timestamp="20260516T000000Z",
        rebuild_stock=False,
    )

    assert cache_dir.exists()
    assert (cache_dir / PLAYER_VALUES_FILE).exists()
    assert (cache_dir / PLAYER_TRENDS_FILE).exists()
    assert manifest["snapshot"]["derived_files"]["player_values"]["rows"] > 0
    assert manifest["snapshot"]["derived_files"]["player_trends"]["rows"] > 0


def test_snapshot_status_lists_snapshot_outputs(tmp_path):
    build_snapshot(
        source="static",
        cache_root=tmp_path,
        timestamp="20260516T000000Z",
        rebuild_stock=False,
    )

    status = snapshot_status(tmp_path)

    assert len(status) == 1
    assert bool(status.iloc[0]["Player_Trends"])
    assert bool(status.iloc[0]["Player_Values"])
