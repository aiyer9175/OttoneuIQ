import os
import sys


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
sys.path.insert(0, os.path.abspath(DATA_DIR))

from historical_snapshots import (
    DEFAULT_BASELINE_TIMESTAMP,
    DEFAULT_OPENING_2026_TIMESTAMP,
    write_end_2025_mlb_snapshot,
    write_historical_value_snapshot,
)
from snapshots import PLAYER_TRENDS_FILE
from trend_history import build_trend_history


def test_write_end_2025_mlb_snapshot_creates_baseline_rows(tmp_path):
    cache_dir, manifest = write_end_2025_mlb_snapshot(cache_root=tmp_path)

    history = build_trend_history(tmp_path)

    assert cache_dir.name == DEFAULT_BASELINE_TIMESTAMP
    assert (cache_dir / PLAYER_TRENDS_FILE).exists()
    assert manifest["snapshot"]["source"] == "preseason_fangraphs_auction_exports"
    assert len(history) > 0
    assert set(history["Trend_Label"]) == {"End-2025 Baseline"}


def test_write_opening_day_snapshot_creates_labeled_checkpoint(tmp_path):
    cache_dir, manifest = write_historical_value_snapshot(
        cache_root=tmp_path,
        timestamp=DEFAULT_OPENING_2026_TIMESTAMP,
        label="Opening Day 2026 Projection",
        notes="2026 preseason FanGraphs projection dollar baseline",
        source_label="2026_opening_day_fangraphs_projection_exports",
    )

    history = build_trend_history(tmp_path)

    assert cache_dir.name == DEFAULT_OPENING_2026_TIMESTAMP
    assert manifest["snapshot"]["source"] == "2026_opening_day_fangraphs_projection_exports"
    assert set(history["Trend_Label"]) == {"Opening Day 2026 Projection"}
