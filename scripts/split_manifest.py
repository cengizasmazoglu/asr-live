#!/usr/bin/env python3
import argparse, json, random
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("work/manifest.jsonl"))
    ap.add_argument("--out_dir", type=Path, default=Path("work/splits"))
    ap.add_argument("--val_ratio", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.manifest.exists():
        raise SystemExit(f"Missing manifest: {args.manifest}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "train.jsonl"
    val_path   = args.out_dir / "val.jsonl"

    random.seed(args.seed)

    # Group by source stem (everything before "_seg_") to avoid leakage
    groups = {}
    for line in args.manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        stem = Path(r["audio"]).stem.split("_seg_")[0]
        groups.setdefault(stem, []).append(r)

    stems = list(groups.keys())
    random.shuffle(stems)

    n_val = max(1, int(len(stems) * args.val_ratio))
    val_stems = set(stems[:n_val])

    train_rows, val_rows = [], []
    for stem, rows in groups.items():
        (val_rows if stem in val_stems else train_rows).extend(rows)

    train_path.write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in train_rows) + "\n",
        encoding="utf-8",
    )
    val_path.write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in val_rows) + "\n",
        encoding="utf-8",
    )

    print("MP3 sources:", len(stems))
    print("VAL sources:", len(val_stems))
    print("Train segments:", len(train_rows))
    print("Val segments:", len(val_rows))
    print("Wrote:", train_path)
    print("Wrote:", val_path)

if __name__ == "__main__":
    main()

