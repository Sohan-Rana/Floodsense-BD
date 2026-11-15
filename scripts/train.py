"""Convenience wrapper: ``python scripts/train.py`` trains and saves the model."""

import sys

from floodsense.cli import main

if __name__ == "__main__":
    sys.exit(main(["train", *sys.argv[1:]]))
