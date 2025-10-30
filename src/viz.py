import argparse
import json
import re
from pathlib import Path
from PIL import Image, ImageDraw


def _norm_bbox(b):
    if b is None:
        return None
    if isinstance(b, dict):
        xs = [b.get("x1"), b.get("x_min"), b.get("left")]
        ys = [b.get("y1"), b.get("y_min"), b.get("top")]
        xe = [b.get("x2"), b.get("x_max"), b.get("right")]
        ye = [b.get("y2"), b.get("y_max"), b.get("bottom")]
        if any(v is None for v in xs + ys + xe + ye):
            return None
        b = [xs[0], ys[0], xe[0], ye[0]]
    if isinstance(b, (list, tuple)) and len(b) == 4:
        try:
            return [int(round(float(v))) for v in b]
        except Exception:
            return None
    return None


def _parse_llm_boxes(answer_text):
    if not isinstance(answer_text, str):
        return []
    m = re.search(r"\{.*\}", answer_text, flags=re.S)
    if not m:
        return []
    try:
        j = json.loads(m.group(0))
    except Exception:
        return []
    boxes = []
    for e in j.get("evidence", []):
        b = e.get("bbox") or e.get("bbox_2d") or e.get("box")
        b = _norm_bbox(b)
        if b:
            boxes.append(b)
    return boxes


def draw(image_path: str, bboxes: list, save_path: str, title: str | None = None):
    im = Image.open(image_path).convert("RGB")
    W, H = im.size
    dr = ImageDraw.Draw(im, "RGBA")
    for i, b in enumerate(bboxes):
        x1, y1, x2, y2 = b
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W - 1, x2), min(H - 1, y2)
        dr.rectangle([x1, y1, x2, y2], outline=(255, 0, 0, 255), width=3)
        dr.rectangle([x1, max(0, y1 - 22), x1 + 120, y1], fill=(255, 0, 0, 120))
        dr.text((x1 + 4, max(0, y1 - 20)), f"ev#{i + 1}", fill=(255, 255, 255, 255))
    if title:
        dr.text((8, 8), title, fill=(0, 0, 0, 255))
    Path(Path(save_path).parent).mkdir(parents=True, exist_ok=True)
    im.save(save_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa_json", required=True)
    ap.add_argument("--out", default="runs/figs/overlay.png")
    args = ap.parse_args()
    obj = json.loads(Path(args.qa_json).read_text(encoding="utf-8"))
    image = obj["image"]
    ev_llm = _parse_llm_boxes(obj.get("answer", ""))
    ev_retr = []
    for e in obj.get("evidence", []):
        if isinstance(e, dict) and e.get("image") == image:
            b = _norm_bbox(e.get("bbox"))
            if b:
                ev_retr.append(b)
    bboxes = ev_retr + ev_llm
    draw(image, bboxes, args.out, title="Evidence Overlay")
    print(f"Saved -> {args.out}  (retr={len(ev_retr)}, llm={len(ev_llm)})")


if __name__ == "__main__":
    main()