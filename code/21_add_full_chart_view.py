import argparse
import glob
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Add a full technical chart image path as another pure image view.")
    parser.add_argument("--manifest_csv", required=True)
    parser.add_argument("--full_image_root", required=True, help="ImageFolder root containing train/val/test/class/*.png")
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--view_name", default="full")
    args = parser.parse_args()

    df = pd.read_csv(args.manifest_csv)
    root = Path(args.full_image_root)
    paths = {}
    for p in root.glob("*/*/*.png"):
        paths[p.stem] = str(p)
    out_col = f"{args.view_name}_path"
    missing = []
    values = []
    for sid in df["sample_id"]:
        path = paths.get(str(sid))
        if path is None:
            matches = glob.glob(str(root / "*" / "*" / f"{sid}.png"))
            path = matches[0] if matches else None
        if path is None:
            missing.append(sid)
            values.append("")
        else:
            values.append(path)
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} full-view images, first={missing[:5]}")
    df[out_col] = values
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print(f">>> wrote {args.out_csv} rows={len(df)} view={args.view_name}", flush=True)


if __name__ == "__main__":
    main()
