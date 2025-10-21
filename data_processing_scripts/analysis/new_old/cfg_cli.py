#!/usr/bin/env python3
import os, sys, json, argparse
from pathlib import Path

try:
    import yaml
except Exception:
    print("[ERROR] Requires PyYAML: pip install pyyaml", file=sys.stderr); sys.exit(2)

def _abs(p: str) -> str:
    return str(Path(p).expanduser().resolve())

def load_cfg(path: str):
    p = Path(path)
    if not p.exists():
        # try relative to repo root / $HOME
        alt = Path.home() / path
        if alt.exists():
            p = alt
        else:
            print(f"[ERROR] config not found: {path}", file=sys.stderr); sys.exit(1)
    with open(p, "r") as f:
        return yaml.safe_load(f)

def get_by_dot(d, key):
    cur = d
    for part in key.split("."):
        if isinstance(cur, list):
            try:
                idx = int(part)
                cur = cur[idx]
            except Exception:
                raise KeyError(key)
        else:
            if part not in cur:
                raise KeyError(key)
            cur = cur[part]
    return cur

def main():
    ap = argparse.ArgumentParser(description="Config reader (dot-path).")
    ap.add_argument("--config", default="analysis/config.yaml")
    ap.add_argument("--get", help="Return value at dot-path (JSON).")
    ap.add_argument("--getcsv", help="Return list at dot-path as comma-separated.")
    args = ap.parse_args()

    cfg = load_cfg(args.config)

    if args.get:
        val = get_by_dot(cfg, args.get)
        print(json.dumps(val, ensure_ascii=False))
        return

    if args.getcsv:
        val = get_by_dot(cfg, args.getcsv)
        if val is None:
            print("")
            return
        if not isinstance(val, (list, tuple)):
            print(str(val))
            return
        print(",".join(str(x) for x in val))
        return

    ap.print_help()

if __name__ == "__main__":
    main()
