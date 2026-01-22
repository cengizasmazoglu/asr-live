#!/usr/bin/env python3
import argparse, json
from pathlib import Path

def rewrite(in_path: Path, out_path: Path, segments_dir: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            fname = Path(r["audio"]).name
            r["audio"] = str(Path(segments_dir) / fname)
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_in", type=Path, default=Path("work/splits_clean/train.jsonl"))
    ap.add_argument("--val_in", type=Path, default=Path("work/splits_clean/val.jsonl"))
    ap.add_argument("--out_dir", type=Path, default=Path("work/splits_runpod"))
    ap.add_argument("--segments_dir", default="/workspace/asr-live/work/segments")
    args = ap.parse_args()

    ntr = rewrite(args.train_in, args.out_dir / "train.jsonl", args.segments_dir)
    nva = rewrite(args.val_in,   args.out_dir / "val.jsonl",   args.segments_dir)

    print("Wrote:", args.out_dir / "train.jsonl", "rows:", ntr)
    print("Wrote:", args.out_dir / "val.jsonl",   "rows:", nva)
    print("Segments dir in manifests:", args.segments_dir)

if __name__ == "__main__":
    main()
