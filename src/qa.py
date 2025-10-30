import argparse
import json
import re
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq


def load_qwen(model_dir: str):
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()
    torch.set_grad_enabled(False)
    return processor, model


def topk(db_dir: Path, query: str, k: int = 5, embed_model: str = "sentence-transformers/all-MiniLM-L6-v2", target_image: str | None = None, search_k: int = 64):
    index = faiss.read_index(str(db_dir / "chunks.faiss"))
    metas = [json.loads(x) for x in Path(db_dir / "meta.jsonl").read_text(encoding="utf-8").splitlines()]
    sbert = SentenceTransformer(embed_model)
    qv = sbert.encode(query, normalize_embeddings=True).astype("float32")
    D, I = index.search(qv.reshape(1, -1), search_k)
    items: list[dict] = []
    for idx, score in zip(I[0], D[0]):
        m = metas[idx]
        if target_image and m["image"] != target_image:
            continue
        m["score"] = float(score)
        items.append(m)
        if len(items) >= k:
            break
    return items


PHONE_RE = re.compile(r"\b(?:\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}|\d{7,})\b")
MONTH_TOKEN = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t)?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
DATE_RE = re.compile(rf"\b(\d{{1,2}}[/-]\d{{1,2}}[/-]\d{{2,4}}|{MONTH_TOKEN}\s+\d{{1,2}},?\s*\d{{2,4}})\b", re.I)
MONTH = {
    "jan": "01",
    "january": "01",
    "feb": "02",
    "february": "02",
    "mar": "03",
    "march": "03",
    "apr": "04",
    "april": "04",
    "may": "05",
    "jun": "06",
    "june": "06",
    "jul": "07",
    "july": "07",
    "aug": "08",
    "august": "08",
    "sep": "09",
    "sept": "09",
    "september": "09",
    "oct": "10",
    "october": "10",
    "nov": "11",
    "november": "11",
    "dec": "12",
    "december": "12",
}


def _load_same_page_chunks(extract_dir: Path, image_path: str):
    for name in ["funsd_test.jsonl", "funsd_train.jsonl", "chartqa.jsonl"]:
        p = Path(extract_dir) / name
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                if ex.get("image") == image_path:
                    return ex.get("chunks", [])
    return []


def _fallback_same_page_chunks(extract_dir: Path, image_path: str, question: str, topn: int = 3):
    chunks = _load_same_page_chunks(extract_dir, image_path)
    cands: list[dict] = []
    q = (question or "").lower()
    for ch in chunks:
        txt = (ch.get("text") or "")
        t = txt.lower()
        score = 0.0
        if "fax" in q or "phone" in q or "telephone" in q:
            if PHONE_RE.search(txt):
                score += 4
            if "fax" in t or "phone" in t:
                score += 1
        if "date" in q:
            if DATE_RE.search(txt):
                score += 4
            if "date" in t:
                score += 1
        if "page" in q:
            if "page" in t or "pages" in t:
                score += 2
            if re.search(r"\b\d+\b", txt):
                score += 1
        if re.search(r"\d", txt):
            score += 0.3
        if len(t) <= 80:
            score += 0.2
        if score > 0:
            cands.append({"image": image_path, "text": txt, "bbox": ch.get("bbox", [0, 0, 0, 0]), "score": score})
    if not cands:
        try:
            W, H = Image.open(image_path).convert("RGB").size
        except Exception:
            W, H = 1000, 1400
        cands = [{"image": image_path, "text": "page", "bbox": [0, 0, W, H], "score": 0.1}]
    cands.sort(key=lambda x: x["score"], reverse=True)
    return cands[:topn]


def _norm_date(s: str):
    if not s:
        return None
    t = s.strip().lower().replace(",", " ").replace("  ", " ")
    m = re.search(rf"{MONTH_TOKEN}\s+(\d{{1,2}}).*?(\d{{2,4}})", t, re.I)
    if m:
        mm = MONTH[m.group(1).lower()]
        dd = int(m.group(2))
        yy = int(m.group(3))
        yyyy = yy if yy > 99 else (1900 + yy if yy >= 30 else 2000 + yy)
        return f"{int(mm):02d}/{dd:02d}/{yyyy:04d}"
    m = re.search(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", t)
    if m:
        mm, dd, yy = int(m.group(1)), int(m.group(2)), int(m.group(3))
        yyyy = yy if yy > 99 else (1900 + yy if yy >= 30 else 2000 + yy)
        return f"{mm:02d}/{dd:02d}/{yyyy:04d}"
    return None


def _date_from_evidence(evid_items):
    for ev in evid_items or []:
        txt = (ev.get("text") or "")
        m = DATE_RE.search(txt)
        if m:
            norm = _norm_date(m.group(0))
            if norm:
                return norm, ev.get("bbox")
    return None, None


def _digits_only(s: str) -> str:
    return re.sub(r"\D", "", s or "")


def _date_candidates_from_same_page(extract_dir: Path, image_path: str, topn: int = 3):
    chunks = _load_same_page_chunks(extract_dir, image_path) or []
    cands: list[dict] = []
    for ch in chunks:
        t = str(ch.get("text", ""))
        if DATE_RE.search(t):
            cands.append({"image": image_path, "text": t, "bbox": ch.get("bbox", [0, 0, 0, 0]), "score": 10.0})
    def _line_id(b):
        return int((b[1] if b else 0) // 18)
    rows: dict = {}
    for ch in chunks:
        b = ch.get("bbox", [0, 0, 0, 0])
        rows.setdefault(_line_id(b), []).append(ch)
    for rid, row in rows.items():
        row = sorted(row, key=lambda x: x.get("bbox", [0, 0, 0, 0])[0])
        row_texts = [str(x.get("text", "")) for x in row]
        if any("date" in t.lower() for t in row_texts):
            idxs = [i for i, t in enumerate(row_texts) if "date" in t.lower()]
            start = min(idxs) + 1 if idxs else 0
            right = row[start : start + 6]
            picked = []
            for ch in right:
                tt = str(ch.get("text", ""))
                if DATE_RE.search(tt) or re.search(rf"{MONTH_TOKEN}|\d", tt, re.I):
                    picked.append(ch.get("bbox", [0, 0, 0, 0]))
            if picked:
                bb = picked[0]
                for b in picked[1:]:
                    bb = [min(bb[0], b[0]), min(bb[1], b[1]), max(bb[2], b[2]), max(bb[3], b[3])]
                cands.append({"image": image_path, "text": " ".join(row_texts[start : start + 6])[:120], "bbox": bb, "score": 9.5})
    uniq: dict = {}
    for e in cands:
        uniq.setdefault(tuple(e["bbox"]), e)
    cands = list(uniq.values())
    cands.sort(key=lambda x: (-x["score"], x["bbox"][1], x["bbox"][0]))
    return cands[:topn]


def _boost_date_evidence(extract_dir: Path, image_path: str, evid_items: list, topn: int = 3):
    boost = _date_candidates_from_same_page(extract_dir, image_path, topn=topn)
    have = {tuple(e.get("bbox", [0, 0, 0, 0])) for e in evid_items}
    merged: list = []
    for e in boost:
        if tuple(e.get("bbox", [0, 0, 0, 0])) not in have:
            merged.append(e)
    merged.extend(evid_items)
    return merged


def ask_with_evidence(processor, model, image_path: str, question: str, evid_items: list, max_new_tokens: int = 180):
    img = Image.open(image_path).convert("RGB")
    if evid_items:
        lines = []
        for i, e in enumerate(evid_items):
            raw_text = (e.get("text") or "")
            safe_text = raw_text.replace("\n", " ")[:160]
            bbox = e.get("bbox", [0, 0, 0, 0])
            lines.append(f"- evidence#{i+1} bbox={bbox} text={safe_text}")
        ev_desc = "\n".join(lines)
    else:
        ev_desc = "(no evidence; rely on the image content)"
    prompt = (
        "You are a document QA assistant. Answer using ONLY the given page image; "
        "the evidence chunks (text+bbox) are hints from the same page.\n"
        "Return STRICT JSON only:\n"
        "{"
        "  'answer': 'str|number|date|unknown',"
        "  'value_bbox': [x1,y1,x2,y2],"
        "  'evidence': [{'bbox':[x1,y1,x2,y2]}],"
        "  'confidence': 0-1"
        "}\n"
        "Rules:\n"
        "- If a concrete date is visible, return it normalized to MM/DD/YYYY.\n"
        "- For phone/fax, return only digits (no spaces or hyphens).\n"
        "- If unsure, set 'answer'='unknown'.\n"
        "- Never output placeholders like 'MM/DD/YYYY' unless 'answer'='unknown'.\n"
        "- Make 'value_bbox' tightly cover the digits/date.\n"
        f"Question: {question}\nEvidence:\n{ev_desc}\n"
    )
    msgs = [
        {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}
    ]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt").to(model.device)
    out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen = processor.batch_decode(out_ids[:, inputs["input_ids"].shape[-1] :], skip_special_tokens=True)[0]
    m = re.search(r"\{.*\}", gen, flags=re.S)
    resp = gen if m is None else m.group(0)
    try:
        j = json.loads(resp)
    except Exception:
        try:
            j = json.loads(resp.replace("'", '"'))
        except Exception:
            j = {"answer": resp}
    ans = (j.get("answer") or "").strip()
    qlow = (question or "").lower()
    if "date" in qlow:
        norm = _norm_date(ans)
        if ans.upper() == "MM/DD/YYYY" or norm is None:
            ev_date, ev_bbox = _date_from_evidence(evid_items)
            if ev_date:
                j["answer"] = ev_date
                if not (isinstance(j.get("value_bbox"), list) and len(j["value_bbox"]) == 4):
                    j["value_bbox"] = ev_bbox or j.get("value_bbox")
            else:
                j["answer"] = "unknown"
    if "phone" in qlow or "fax" in qlow or "telephone" in qlow:
        d = _digits_only(ans)
        if d:
            j["answer"] = d
        else:
            for ev in evid_items or []:
                dd = _digits_only(ev.get("text") or "")
                if len(dd) >= 7:
                    j["answer"] = dd
                    if not (isinstance(j.get("value_bbox"), list) and len(j["value_bbox"]) == 4):
                        j["value_bbox"] = ev.get("bbox")
                    break
            if not j.get("answer"):
                j["answer"] = "unknown"
    vb = j.get("value_bbox")
    if isinstance(vb, list) and len(vb) == 1 and isinstance(vb[0], list) and len(vb[0]) == 4:
        j["value_bbox"] = vb[0]
    if not (isinstance(j.get("value_bbox"), list) and len(j["value_bbox"]) == 4) and evid_items:
        j["value_bbox"] = evid_items[0].get("bbox", [0, 0, 0, 0])
    return json.dumps(j, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db_dir", default="runs/index")
    ap.add_argument("--extract_dir", default="runs/extract")
    ap.add_argument("--model_dir", default="models/qwen")
    ap.add_argument("--image", required=True)
    ap.add_argument("--q", required=True)
    ap.add_argument("--out", default="runs/qa/answer.json")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--max_evidence", type=int, default=4)
    args = ap.parse_args()
    Path(Path(args.out).parent).mkdir(parents=True, exist_ok=True)
    evid = topk(Path(args.db_dir), args.q, k=max(args.k, 5), target_image=args.image)
    if len(evid) == 0:
        evid = _fallback_same_page_chunks(Path(args.extract_dir), args.image, args.q, topn=min(args.k, 3))
    if "date" in args.q.lower():
        evid = _boost_date_evidence(Path(args.extract_dir), args.image, evid, topn=3)
    evid = evid[: max(1, args.max_evidence)]
    proc, model = load_qwen(args.model_dir)
    ans = ask_with_evidence(proc, model, args.image, args.q, evid, max_new_tokens=args.max_new_tokens)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"image": args.image, "question": args.q, "answer": ans, "evidence": evid}, f, ensure_ascii=False, indent=2)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()