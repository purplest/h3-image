#!/usr/bin/env python3
"""Range-fetch selected tensors from Hugging Face safetensors into a small
local file (SPEC §6.6). No full shard download.

Usage: fetch_tensors.py REPO SHARD OUTPUT NAME...
  REPO   e.g. MiniMaxAI/MiniMax-H3-Image
  SHARD  e.g. transformer/model-00001-of-00012.safetensors,
         or '-' to auto-resolve each name via transformer/model.safetensors.index.json
  OUTPUT local .safetensors path (parent dirs created)
  NAME   tensor names to extract (dtype/shape preserved, bytes verbatim)

Retries each range request 8 times with exponential backoff.
"""
import json
import os
import struct
import sys
import time
import urllib.request

BASE = "https://huggingface.co/{repo}/resolve/main/{path}"
UA = "h3-image-fetch/0.1"


def fetch_range(url, start, end, tries=8):
    last = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(
                url, headers={"Range": f"bytes={start}-{end}", "User-Agent": UA})
            with urllib.request.urlopen(req, timeout=300) as r:
                data = r.read()
            want = end - start + 1
            if len(data) != want:
                raise IOError(f"short read: {len(data)} of {want} bytes")
            return data
        except Exception as e:  # noqa: BLE001 - retry any transport failure
            last = e
            if attempt == tries - 1:
                raise
            time.sleep(min(2 ** attempt, 60))
    raise last


def fetch_all(url, tries=8):
    last = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            if attempt == tries - 1:
                raise
            time.sleep(min(2 ** attempt, 60))
    raise last


def read_header(url):
    n = struct.unpack("<Q", fetch_range(url, 0, 7))[0]
    if n > 256 * 1024 * 1024:
        raise ValueError(f"implausible safetensors header size {n}")
    header = json.loads(fetch_range(url, 8, 8 + n - 1))
    return header, 8 + n


def main():
    if len(sys.argv) < 5:
        sys.stderr.write(__doc__)
        return 2
    repo, shard, output = sys.argv[1:4]
    names = sys.argv[4:]

    # name -> shard path
    if shard == "-":
        index_url = BASE.format(
            repo=repo, path="transformer/model.safetensors.index.json")
        index = json.loads(fetch_all(index_url))
        weight_map = index["weight_map"]
        missing = [n for n in names if n not in weight_map]
        if missing:
            sys.stderr.write(f"not in the transformer index: {missing}\n")
            return 1
        plan = {}
        for n in names:
            plan.setdefault("transformer/" + weight_map[n], []).append(n)
    else:
        plan = {shard: names}

    tensors = {}
    blobs = []
    offset = 0
    for shard_path, shard_names in plan.items():
        url = BASE.format(repo=repo, path=shard_path)
        header, data_start = read_header(url)
        for name in shard_names:
            if name not in header:
                sys.stderr.write(f"{name} not in {shard_path}\n")
                return 1
            t = header[name]
            b, e = t["data_offsets"]
            data = fetch_range(url, data_start + b, data_start + e - 1)
            if len(data) != e - b:
                sys.stderr.write(f"{name}: byte count mismatch\n")
                return 1
            tensors[name] = {
                "dtype": t["dtype"],
                "shape": t["shape"],
                "data_offsets": [offset, offset + len(data)],
            }
            blobs.append(data)
            offset += len(data)
            sys.stderr.write(f"fetched {name} ({len(data)} bytes)\n")

    out_header = json.dumps(tensors, separators=(",", ":")).encode("utf-8")
    parent = os.path.dirname(output)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output, "wb") as f:
        f.write(struct.pack("<Q", len(out_header)))
        f.write(out_header)
        for blob in blobs:
            f.write(blob)
    sys.stderr.write(f"wrote {output} ({len(tensors)} tensors, {offset} bytes)\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
