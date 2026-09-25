"""Install an HPA-RMVPE checkpoint by hand, or fetch it ahead of first use.

The HuggingFace files (AnhP/HPA-RMVPE) are 203 MB training checkpoints, of which 68 MB
is the model. With a source file this checks its sha256 against the published one,
keeps checkpoint["model"] and writes rvc/models/predictors/hpa-rmvpe-<variant>.pt with
its manifest. Without one it downloads the pinned revision first, which is what the
first use of the method would otherwise do.

    python tools/strip_hpa_rmvpe.py 76000 [model_76000.pt]
    python tools/strip_hpa_rmvpe.py 112000 [model_112000.pt]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rvc.lib.predictors.hpa_rmvpe.weights import VARIANTS, download, strip_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variant", choices=list(VARIANTS))
    parser.add_argument("source", nargs="?", help="a downloaded checkpoint; omit to download it")
    args = parser.parse_args()
    manifest = strip_checkpoint(args.variant, args.source) if args.source else download(args.variant)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
