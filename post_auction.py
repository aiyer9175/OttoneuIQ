import os
import sys


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
if DATA_DIR not in sys.path:
    sys.path.insert(0, DATA_DIR)

from data.post_auction import main


if __name__ == "__main__":
    main()
