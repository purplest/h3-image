#!/usr/bin/env python3
"""Extract selected tensors from LOCAL safetensors shards into a small file
(the offline counterpart of fetch_tensors.py — for machines that already
have the official model).

Usage: extract_tensors.py SHARD_DIR OUTPUT NAME...
  SHARD_DIR  directory with model.safetensors.index.json + shards,
             or a single model.safetensors
  OUTPUT     local .safetensors path (parent dirs created)
  NAME       tensor names (dtype/shape preserved, bytes verbatim)
"""
import json
import os
import struct
import sys


def main():
    if len(sys.argv) < 4:
        sys.stderr.write(__doc__)
        return 2
    shard_dir, output = sys.argv[1:3]
    names = sys.argv[3:]

    index_path = os.path.join(shard_dir, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
        missing = [n for n in names if n not in weight_map]
        if missing:
            sys.stderr.write(f"not in the index: {missing}\n")
            return 1
        plan = {}
        for n in names:
            plan.setdefault(weight_map[n], []).append(n)
    else:
        single = os.path.join(shard_dir, "model.safetensors")
        if not os.path.isfile(single):
            sys.stderr.write(f"no index or model.safetensors in {shard_dir}\n")
            return 1
        plan = {"model.safetensors": names}

    tensors = {}
    blobs = []
    offset = 0
    for shard, shard_names in plan.items():
        path = os.path.join(shard_dir, shard)
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
            data_start = 8 + n
            for name in shard_names:
                if name not in header:
                    sys.stderr.write(f"{name} not in {shard}\n")
                    return 1
                t = header[name]
                b, e = t["data_offsets"]
                f.seek(data_start + b)
                data = f.read(e - b)
                if len(data) != e - b:
                    sys.stderr.write(f"{name}: short read (truncated shard?)\n")
                    return 1
                tensors[name] = {
                    "dtype": t["dtype"],
                    "shape": t["shape"],
                    "data_offsets": [offset, offset + len(data)],
                }
                blobs.append(data)
                offset += len(data)
                sys.stderr.write(f"extracted {name} ({len(data)} bytes)\n")

    parent = os.path.dirname(output)
    if parent:
        os.makedirs(parent, exist_ok=True)
    out_header = json.dumps(tensors, separators=(",", ":")).encode("utf-8")
    with open(output, "wb") as f:
        f.write(struct.pack("<Q", len(out_header)))
        f.write(out_header)
        for blob in blobs:
            f.write(blob)
    sys.stderr.write(f"wrote {output} ({len(tensors)} tensors, {offset} bytes)\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
