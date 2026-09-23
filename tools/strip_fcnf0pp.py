"""Strip the published FCNF0++ training checkpoint down to its weights.

The HuggingFace file (maxrmorrison/fcnf0-plus-plus) is 107 MB, of which 68 MB is
Adam state. This keeps checkpoint["model"] (34 MB), refuses any file whose sha256 is
not the published one, and writes rvc/models/predictors/fcnf0++.manifest.json.
Stripping in place is supported: the output replaces the source atomically.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rvc.lib.predictors.fcnf0pp.weights import DEFAULT_WEIGHT, strip_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", default=str(DEFAULT_WEIGHT))
    parser.add_argument("output", nargs="?", default=str(DEFAULT_WEIGHT))
    args = parser.parse_args()
    print(json.dumps(strip_checkpoint(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
