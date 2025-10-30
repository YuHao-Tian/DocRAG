import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from rapidfuzz import fuzz


NUM_RE = re.compile(r"\d+")
DATE_RE = re.compile(
    r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2},?\s*\d{2,4})\b",
    re.I,
)
PHONE_RE = re.compile(r"\(?\d{3}\)?[\s\-]\d{3}[\s\-]\d{4}")
PAGES_KEYWORDS = ("pages", "including", "cover")


def iou(a: List[int], b: List[int]) -> float:
    if not a or not b:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / max(area, 1e-6)


def norm_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = re.sub(r"[,\s\.\-_/]", "", s.strip()).lower()
    return s


def pick_pred_bbox_from_answer(ans_text: str) -> Optional[List[int]]:
    if not isinstance(ans_text, str):
        return None
    m = re.search(r"\{.*\}", ans_text, flags=re.S)
    if not m:
        return None
    try:
        j = json.loads(m.group(0))
    except Exception:
        return None
    if isinstance(j, dict):
        for k in ("value_bbox", "bbox", "bbox_2d"):
            b = j.get(k)
            if isinstance(b, list) and len(b) == 4:
                return [int(float(x)) for x in b]
        ev = j.get("evidence") or []
        for e in ev:
            b = e.get("bbox") or e.get("bbox_2d")
            if isinstance(b, list) and len(b) == 4:
                return [int(float(x)) for x in b]
    return None


def make_gold_from_chunks(chunks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    gold: Dict[str, Dict[str, Any]] = {k: {"value": None, "bbox": None} for k in ["date", "fax", "phone", "pages"]}
    for ch in chunks:
        t = str(ch.get("text", ""))
        m = DATE_RE.search(t)
        if m:
            gold["date"] = {"value": m.group(0), "bbox": ch.get("bbox")}
            break
    for ch in chunks:
        t = str(ch.get("text", "")).lower()
        pass
    for ch in chunks:
        t = str(ch.get("text", ""))
        m = PHONE_RE.search(t)
        if m and gold["phone"]["value"] is None:
            gold["phone"] = {"value": m.group(0), "bbox": ch.get("bbox")}
        if m and gold["fax"]["value"] is None and "fax" in t.lower():
            gold["fax"] = {"value": m.group(0), "bbox": ch.get("bbox")}
    if gold["fax"]["value"] is None:
        for ch in chunks:
            if "fax" in str(ch.get("text", "")).lower():
                for ch2 in chunks:
                    t2 = str(ch2.get("text", ""))
                    m2 = PHONE_RE.search(t2)
                    if m2:
                        gold["fax"] = {"value": m2.group(0), "bbox": ch2.get("bbox")}
                        break
                break
    cand_idx = -1
    for i, ch in enumerate(chunks):
        low = str(ch.get("text", "")).lower()
        if all(k in low for k in ("page", "cover")) or all(k in low for k in PAGES_KEYWORDS):
            cand_idx = i
            break
    if cand_idx >= 0:
        for j in range(cand_idx, min(cand_idx + 8, len(chunks))):
            m = NUM_RE.search(str(chunks[j].get("text", "")))
            if m:
                gold["pages"] = {"value": m.group(0), "bbox": chunks[j].get("bbox")}
                break
    else:
        for i, ch in enumerate(chunks):
            low = str(ch.get("text", "")).lower()
            if "page" in low:
                for j in range(i, min(i + 8, len(chunks))):
                    m = NUM_RE.search(str(chunks[j].get("text", "")))
                    if m:
                        gold["pages"] = {"value": m.group(0), "bbox": chunks[j].get("bbox")}
                        break
                break
    return gold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract_jsonl", default="runs/extract/funsd_test.jsonl", help="Extraction JSONL file")
    ap.add_argument("--qa_dir", default="runs/qa", help="Directory with QA outputs (JSON files)")
    ap.add_argument("--out", default="runs/eval/funsd_metrics.json", help="Output JSON file for metrics")
    ap.add_argument("--iou_thr", type=float, default=0.5, help="IoU threshold for box matching")
    args = ap.parse_args()
    preds: Dict[str, Dict[str, Any]] = {}
    qa_dir = Path(args.qa_dir)
    if qa_dir.exists():
        for p in qa_dir.glob("*.json"):
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
                ans = obj.get("answer", "")
                bbox = pick_pred_bbox_from_answer(ans)
                preds[obj["image"]] = {"answer": ans, "bbox": bbox}
            except Exception:
                pass
    totals = {k: 0 for k in ["date", "fax", "phone", "pages"]}
    correct_val = {k: 0 for k in totals}
    correct_box = {k: 0 for k in totals}
    with open(args.extract_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            img = ex.get("image")
            chunks = ex.get("chunks", [])
            gold = make_gold_from_chunks(chunks)
            pred = preds.get(img, {"answer": "", "bbox": None})
            ans_txt = pred.get("answer", "")
            ans_match = ans_txt
            m = re.search(r"\{.*\}", ans_txt, flags=re.S)
            if m:
                try:
                    j = json.loads(m.group(0))
                    if isinstance(j, dict) and "answer" in j:
                        ans_match = str(j["answer"])
                except Exception:
                    pass
            box_pred = pred.get("bbox")
            for field, g in gold.items():
                if not g["value"]:
                    continue
                totals[field] += 1
                ok_val = False
                if field in ("pages",):
                    ok_val = (re.search(r"\d+", ans_match or "") and re.search(r"\d+", g["value"])) and (
                        re.search(r"\d+", ans_match).group(0) == re.search(r"\d+", g["value"]).group(0)
                    )
                elif field in ("phone", "fax"):
                    a = re.sub(r"\D", "", ans_match or "")
                    b = re.sub(r"\D", "", g["value"])
                    if len(a) >= 7 and len(b) >= 7:
                        ok_val = a[-7:] == b[-7:] or a[-10:] == b[-10:]
                elif field in ("date",):
                    ok_val = fuzz.token_set_ratio(ans_match, g["value"]) >= 80
                else:
                    ok_val = fuzz.token_set_ratio(ans_match, g["value"]) >= 80
                if ok_val:
                    correct_val[field] += 1
                if box_pred and g["bbox"]:
                    if iou(box_pred, g["bbox"]) >= args.iou_thr:
                        correct_box[field] += 1
    out = {
        "N": totals,
        "Value_Accuracy": {k: round(correct_val[k] / max(1, totals[k]), 4) for k in totals},
        f"Box_IoU@{args.iou_thr:.2f}": {k: round(correct_box[k] / max(1, totals[k]), 4) for k in totals},
    }
    Path(Path(args.out).parent).mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()