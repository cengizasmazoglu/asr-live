import argparse, json, subprocess
from pathlib import Path
from urllib.parse import urlparse, unquote

import srt  # pip install srt

def sh(cmd):
    subprocess.run(cmd, check=True)

def norm_name(x: str) -> str:
    # dosya eşleştirmede tolerans: boşluk/altçizgi/çoklu space
    x = x.lower().replace("_", " ").replace("%20", " ")
    while "  " in x:
        x = x.replace("  ", " ")
    return x.strip()

def find_mp3(mp3_dir: Path, url_basename: str) -> Path | None:
    # 1) exact
    p = mp3_dir / url_basename
    if p.exists():
        return p

    # 2) normalized match
    target = norm_name(url_basename)
    for cand in mp3_dir.glob("*.mp3"):
        if norm_name(cand.name) == target:
            return cand

    # 3) loose contains (son çare)
    for cand in mp3_dir.glob("*.mp3"):
        if target in norm_name(cand.name) or norm_name(cand.name) in target:
            return cand
    return None

def mp3_to_wav16k(mp3: Path, wav: Path):
    wav.parent.mkdir(parents=True, exist_ok=True)
    sh([
        "ffmpeg","-y","-loglevel","error",
        "-i", str(mp3),
        "-ac","1","-ar","16000","-vn",
        str(wav)
    ])

def cut_segment(wav: Path, out_wav: Path, start: float, end: float):
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sh([
        "ffmpeg","-y","-loglevel","error",
        "-ss", f"{start:.3f}",
        "-to", f"{end:.3f}",
        "-i", str(wav),
        "-ac","1","-ar","16000",
        str(out_wav)
    ])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mp3_dir", type=Path, required=True)
    ap.add_argument("--srt_dir", type=Path, required=True)
    ap.add_argument("--urls", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("work"))
    ap.add_argument("--pad_ms", type=int, default=200)
    ap.add_argument("--min_sec", type=float, default=2.0)
    ap.add_argument("--max_sec", type=float, default=30.0)
    args = ap.parse_args()

    wav_dir = args.out / "wav16k"
    seg_dir = args.out / "segments"
    manifest = args.out / "manifest.jsonl"
    args.out.mkdir(parents=True, exist_ok=True)

    urls = [line.strip() for line in args.urls.read_text(encoding="utf-8").splitlines() if line.strip()]

    with manifest.open("w", encoding="utf-8") as mf:
        for i, url in enumerate(urls, start=1):
            url_basename = unquote(Path(urlparse(url).path).name)  # URL'den mp3 adı
            mp3 = find_mp3(args.mp3_dir, url_basename)
            srt_path = args.srt_dir / f"subtitle_{i}.srt"

            if mp3 is None:
                print(f"[SKIP] mp3 not found for #{i}: {url_basename}")
                continue
            if not srt_path.exists():
                print(f"[SKIP] srt not found: {srt_path}")
                continue

            wav = wav_dir / (mp3.stem + "_16k.wav")
            if not wav.exists():
                mp3_to_wav16k(mp3, wav)

            subs = list(srt.parse(srt_path.read_text(encoding="utf-8")))
            seg_idx = 0

            for sub in subs:
                text = (sub.content or "").replace("\n", " ").strip()
                if not text:
                    continue

                start = sub.start.total_seconds() - args.pad_ms/1000.0
                end   = sub.end.total_seconds() + args.pad_ms/1000.0
                start = max(0.0, start)
                dur = end - start

                if dur < args.min_sec or dur > args.max_sec:
                    continue

                seg_idx += 1
                out_wav = seg_dir / f"{mp3.stem}_seg_{seg_idx:06d}.wav"
                cut_segment(wav, out_wav, start, end)

                mf.write(json.dumps({"audio": str(out_wav), "text": text}, ensure_ascii=False) + "\n")

            print(f"[OK] #{i} {mp3.name} -> {seg_idx} segments")

    print("DONE:", manifest)

if __name__ == "__main__":
    main()

