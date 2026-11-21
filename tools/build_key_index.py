"""Build a key/value index next to a model's ordinary one (rvc/lib/key_index.py).

    env314\\Scripts\\python.exe tools\\build_key_index.py <model folder name> [--key-embedder NAME]

Writes logs/<model>/<model>_key-spin-v2.index (+ .index.kv.json, .values.npy). Pick it in
the index dropdown of the inference or realtime tab instead of the ordinary index: the
search then uses the key embedder's features of the source and the model gets its own
training features, as before. Reads the folder's sliced_audios_16k/ and extracted/.

For an embedder that is not registered, give the folder tools/spin/export.py wrote and a
name for the files: --key-embedder <name> --key-embedder-path <folder>, which writes
<model>_key-<name>.index. The folder must stay where it is: the index reads it from there.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rvc.lib.key_index import DEFAULT_KEY_EMBEDDER, build_key_index  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("model")
    parser.add_argument("--key-embedder", default=DEFAULT_KEY_EMBEDDER)
    parser.add_argument("--key-embedder-path")
    parser.add_argument("--key-output-layer", type=int, default=None)
    args = parser.parse_args()
    folder = os.path.join("logs", args.model)
    if not os.path.isdir(folder):
        raise SystemExit(f"No such model folder: {folder}")
    path, spec = build_key_index(folder, args.key_embedder, args.key_output_layer,
                                 key_embedder_path=args.key_embedder_path)
    print(path)
    print(json.dumps(spec, indent=1))


if __name__ == "__main__":
    main()
