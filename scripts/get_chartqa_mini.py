from datasets import load_dataset, DownloadConfig
from pathlib import Path
from PIL import Image
import json
import random
import io
import sys

OUT = Path("data/chartqa_mini")
(OUT / "images").mkdir(parents=True, exist_ok=True)
dcfg = DownloadConfig(max_retries=10, resume_download=False)


def save_image(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, Image.Image):
        obj.convert("RGB").save(path)
    elif isinstance(obj, dict) and "path" in obj:
        Image.open(obj["path"]).convert("RGB").save(path)
    elif isinstance(obj, (bytes, bytearray)):
        Image.open(io.BytesIO(obj)).convert("RGB").save(path)
    else:
        raise ValueError(f"Unsupported image type: {type(obj)}")


def pick(d, *keys, default=""):
    for k in keys:
        v = d.get(k)
        if v is not None and str(v).strip() != "":
            return v
    return default


use_split = None
for split_name in ["validation", "val", "train", "test"]:
    try:
        ds = load_dataset("HuggingFaceM4/ChartQA", split=split_name, streaming=True, download_config=dcfg)
        use_split = split_name
        break
    except Exception:
        continue
else:
    print("Cannot open any split from ChartQA", file=sys.stderr)
    sys.exit(1)

N = 200
buf = []
random.seed(42)
count = 0
for ex in ds:
    ex = dict(ex)
    q = pick(ex, "question", "query", "Query")
    a = pick(ex, "answer", "label", "final_answer")
    if not q or not a:
        continue
    count += 1
    item = {"ex": ex, "q": str(q), "a": str(a)}
    if len(buf) < N:
        buf.append(item)
    else:
        j = random.randint(1, count)
        if j <= N:
            buf[j - 1] = item
    if count >= 10000:
        break

qa_list: list = []
for i, item in enumerate(buf):
    ex, q, a = item["ex"], item["q"], item["a"]
    img_name = f"chart_{i:05d}.png"
    img_out = OUT / "images" / img_name
    save_image(img_out, ex["image"])
    qa_list.append(
        {
            "image": str(img_out),
            "question": q,
            "answer": a,
            "plot_type": ex.get("plot_type") or ex.get("type"),
            "table": ex.get("table"),
        }
    )

with open(OUT / "qa.jsonl", "w", encoding="utf-8") as f:
    for q in qa_list:
        f.write(json.dumps(q, ensure_ascii=False) + "\n")

print(f"ChartQA-mini prepared in {OUT} (kept {len(qa_list)} samples from split='{use_split}')")