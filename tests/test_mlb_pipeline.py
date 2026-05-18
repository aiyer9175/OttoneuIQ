import os
import sys


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
sys.path.insert(0, os.path.abspath(DATA_DIR))

from mlb_pipeline import parse_visible_pipeline_rows


def test_parse_visible_pipeline_rows_from_static_html():
    html = """
    <div>Rank</div><div>Player</div><div>Position</div><div>Current Team</div>
    <div>Current Level</div><div>Current Age</div><div>Bats</div><div>Throws</div>
    <div>1</div><div>Roki Sasaki</div><div>RHP</div><div>Los Angeles Dodgers</div>
    <div>MLB</div><div>24</div><div>R</div><div>R</div>
    <div>2</div><div>Roman Anthony</div><div>OF</div><div>Boston Red Sox</div>
    <div>MLB</div><div>22</div><div>L</div><div>R</div>
    """

    rows = parse_visible_pipeline_rows(html, 2025, "https://example.test")

    assert len(rows) == 2
    assert rows.iloc[0]["Name"] == "Roki Sasaki"
    assert rows.iloc[1]["Rank"] == 2
