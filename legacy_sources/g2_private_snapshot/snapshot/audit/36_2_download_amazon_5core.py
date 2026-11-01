from pathlib import Path
from urllib.request import urlopen, Request
import hashlib
import json
import shutil
import gzip
import ast

ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "Amazon_Sports": {
        "url": (
            "https://snap.stanford.edu/data/amazon/productGraph/"
            "categoryFiles/reviews_Sports_and_Outdoors_5.json.gz"
        ),
        "filename": "reviews_Sports_and_Outdoors_5.json.gz",
    },
    "Amazon_Toys": {
        "url": (
            "https://snap.stanford.edu/data/amazon/productGraph/"
            "categoryFiles/reviews_Toys_and_Games_5.json.gz"
        ),
        "filename": "reviews_Toys_and_Games_5.json.gz",
    },
}

OUT = ROOT / "reproducibility" / "step36"
OUT.mkdir(parents=True, exist_ok=True)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(8 * 1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def inspect_gzip(path):
    line_count = 0
    users = set()
    items = set()

    first_record = None

    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            # Amazon 2014 files are Python-dict-like lines.
            obj = ast.literal_eval(line)

            if first_record is None:
                first_record = obj

            required = {
                "reviewerID",
                "asin",
                "overall",
                "unixReviewTime",
            }

            missing = required - set(obj)

            if missing:
                raise RuntimeError(
                    f"Missing keys {missing} in {path}"
                )

            users.add(str(obj["reviewerID"]))
            items.add(str(obj["asin"]))
            line_count += 1

    return {
        "rows": line_count,
        "users": len(users),
        "items": len(items),
        "first_record_keys": sorted(first_record.keys()),
    }


records = []

print("=" * 78)
print("STEP36.2 AMAZON 2014 5-CORE DOWNLOAD")
print("=" * 78)

for dataset, cfg in DATASETS.items():

    out_dir = ROOT / "data" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    dst = out_dir / cfg["filename"]

    print()
    print("DATASET:", dataset)
    print("URL:", cfg["url"])
    print("TARGET:", dst)

    if dst.exists() and dst.stat().st_size > 0:
        print("DOWNLOAD: SKIP (existing non-empty file)")
    else:
        req = Request(
            cfg["url"],
            headers={
                "User-Agent": "Mozilla/5.0"
            },
        )

        with urlopen(req, timeout=120) as response:
            with dst.open("wb") as f:
                shutil.copyfileobj(response, f)

        print("DOWNLOAD: COMPLETE")

    if not dst.is_file() or dst.stat().st_size == 0:
        raise RuntimeError(f"Download failed: {dst}")

    print("SIZE_MB:", round(dst.stat().st_size / 1024 / 1024, 3))

    digest = sha256(dst)

    print("SHA256:", digest)

    print("VALIDATING GZIP CONTENT...", flush=True)

    stats = inspect_gzip(dst)

    print("ROWS:", stats["rows"])
    print("USERS:", stats["users"])
    print("ITEMS:", stats["items"])
    print("REQUIRED FIELDS: PASS")

    records.append({
        "dataset": dataset,
        "source": "Amazon Reviews 2014 5-core",
        "url": cfg["url"],
        "path": str(dst.relative_to(ROOT)),
        "bytes": dst.stat().st_size,
        "sha256": digest,
        **stats,
    })

manifest = {
    "step": "36.2",
    "source_family": "Amazon Reviews 2014",
    "filter": "published 5-core category files",
    "datasets": records,
}

manifest_path = (
    OUT / "step36_2_amazon_5core_download_manifest.json"
)

manifest_path.write_text(
    json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
    ) + "\n",
    encoding="utf-8",
)

print()
print("=" * 78)
print("SUMMARY")
print("=" * 78)

for r in records:
    print(
        f"{r['dataset']:16s} "
        f"rows={r['rows']:,} "
        f"users={r['users']:,} "
        f"items={r['items']:,}"
    )

print()
print("SAVED:", manifest_path)
print("STEP36.2 STATUS: PASS")
