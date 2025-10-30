from datasets import load_dataset, DownloadConfig
from pathlib import Path
from PIL import Image
import json
import io

OUT_ROOT = Path("data/funsd")
(OUT_ROOT / "train" / "images").mkdir(parents=True, exist_ok=True)
(OUT_ROOT / "test" / "images").mkdir(parents=True, exist_ok=True)
dcfg = DownloadConfig(max_retries=10, resume_download=False)


def save_image_to(path: Path, imgobj):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(imgobj, Image.Image):
        imgobj.convert("RGB").save(path)
    elif isinstance(imgobj, dict) and "path" in imgobj:
        Image.open(imgobj["path"]).convert("RGB").save(path)
    elif isinstance(imgobj, (bytes, bytearray)):
        Image.open(io.BytesIO(imgobj)).convert("RGB").save(path)
    else:
        raise ValueError(f"Unsupported image type: {type(imgobj)}")


def dump_split(split_name: str):
    ds = load_dataset("nielsr/funsd", split=split_name, download_config=dcfg, download_mode="force_redownload")
    out_dir = OUT_ROOT / split_name
    images_dir = out_dir / "images"
    ann_path = out_dir / "labels.jsonl"
    with open(ann_path, "w", encoding="utf-8") as fw:
        for i, ex in enumerate(ds):
            img_name = f"{ex.get('id', str(i)).replace('/', '_')}.png"
            img_out = images_dir / img_name
            save_image_to(img_out, ex["image"])
            rec = {
                "id": ex.get("id", str(i)),
                "image": str(img_out),
                "words": ex.get("words", []),
                "bboxes": ex.get("bboxes", []),
                "ner_tags": ex.get("ner_tags", []),
                "linking": ex.get("linking", []),
            }
            fw.write(json.dumps(rec, ensure_ascii=False) + "\n")


for sp in ["train", "test"]:
    dump_split(sp)
print("FUNSD dataset prepared in data/funsd")