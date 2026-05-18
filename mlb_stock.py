import os
import sys

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
sys.path.insert(0, DATA_DIR)

from mlb_stock import main


if __name__ == "__main__":
    main()
