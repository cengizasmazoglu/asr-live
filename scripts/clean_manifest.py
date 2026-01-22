#!/usr/bin/env python3
import argparse, json, re
from pathlib import Path

PLACEHOLDER_RE = re.compile(r'^\s*(segment-\d+|seg-\d+|segment\s*\d+|\d+)\s*$', re.IGNORECASE)

def is_bad(text: str) -> bool:
    if text is None:
        return True
    t = text.strip()
    if not t:
        return True
    if PLACEHOLDER_RE.match(t):
        return True
    return False

def clean(in_path: Path, out_path: Path):
    bad = 0
    total = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            r = json.loads(line)
            if is_bad(r.get("text", "")):
                bad += 1
                continue
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
    return total, bad

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_in", type=Path, default=Path("work/splits/train.jsonl"))
    ap.add_argument("--val_in", type=Path, default=Path("work/splits/val.jsonl"))
    ap.add_argument("--out_dir", type=Path, default=Path("work/splits_clean"))
    args = ap.parse_args()

    train_out = args.out_dir / "train.jsonl"
    val_out   = args.out_dir / "val.jsonl"

    t_total, t_bad = clean(args.train_in, train_out)
    v_total, v_bad = clean(args.val_in, val_out)

    print("TRAIN:", t_total, "rows,", t_bad, "dropped,", "kept:", t_total - t_bad)
    print("VAL:  ", v_total, "rows,", v_bad, "dropped,", "kept:", v_total - v_bad)
    print("Wrote:", train_out)
    print("Wrote:", val_out)

if __name__ == "__main__":
    main()
