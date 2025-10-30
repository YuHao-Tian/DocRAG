import argparse
import json
import re
import io
from pathlib import Path
from PIL import Image, ImageFile
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

ImageFile.LOAD_TRUNCATED_IMAGES = True

OCR_READY = False
_OCR_READER = None
try:
    import easyocr
    OCR_READY = True
except Exception:
    OCR_READY = False


def get_ocr_reader(lang: str = "en"):
    global _OCR_READER
    if not OCR_READY:
        return None
    if _OCR_READER is None:
        _OCR_READER = easyocr.Reader([lang], gpu=torch.cuda.is_available())
    return _OCR_READER


def load_qwen(model_dir: str):
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_dir,
        trust_remote_code=True,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()
    torch.set_grad_enabled(False)
    return processor, model


def qwen_json_qa(processor, model, img: Image.Image, prompt: str, max_new_tokens: int = 256):
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt").to(model.device)
    out_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
    )
    gen = processor.batch_decode(out_ids[:, inputs["input_ids"].shape[-1] :], skip_special_tokens=True)[0]
    m = re.search(r"\{.*\}", gen, flags=re.S)
    return gen if m is None else m.group(0)


def ensure_dirs(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def ocr_chunks(img_path: str, lang: str = "en"):
    chunks: list[dict] = []
    reader = get_ocr_reader(lang)
    if reader is None:
        return chunks
    res = reader.readtext(str(img_path))
    for it in res:
        try:
            ((x1, y1), (x2, _), (x3, y3), (_, _)) = it[0]
            x_min = int(min(x1, x2, x3))
            y_min = int(min(y1, y3))
            x_max = int(max(x1, x2, x3))
            y_max = int(max(y1, y3))
            chunks.append(
                {
                    "text": it[1],
                    "bbox": [x_min, y_min, x_max, y_max],
                    "conf": float(it[2]),
                }
            )
        except Exception:
            continue
    return chunks


def run_funsd(funsd_root: Path, out_dir: Path, processor=None, model=None, use_ocr: bool = True):
    for split in ["train", "test"]:
        lab = funsd_root / split / "labels.jsonl"
        if not lab.exists():
            continue
        out_jsonl = out_dir / f"funsd_{split}.jsonl"
        with open(lab, "r", encoding="utf-8") as fr, open(out_jsonl, "w", encoding="utf-8") as fw:
            for line in fr:
                ex = json.loads(line)
                img_path = ex["image"]
                img = Image.open(img_path).convert("RGB")
                W, H = img.size
                chunks = [
                    {"text": w, "bbox": b, "conf": 1.0}
                    for w, b in zip(ex.get("words", []), ex.get("bboxes", []))
                ]
                if use_ocr and len(chunks) < 10:
                    chunks += ocr_chunks(img_path, lang="en")
                kv_pred = None
                if processor and model:
                    prompt = (
                        "You are a document information extraction assistant. "
                        "Given a scanned form image, extract up to 15 likely key-value pairs. "
                        "Return a STRICT JSON with the following schema:\n"
                        "{"
                        "  'kv_pairs': ["
                        "    {'key': str, 'value': str, 'key_bbox':[x1,y1,x2,y2], 'value_bbox':[x1,y1,x2,y2], 'confidence': 0-1}"
                        "  ]"
                        "}\n"
                        "Bounding boxes must be pixel coordinates aligned to the input image. "
                        "If a field is not found, omit it. Do not add explanations."
                    )
                    try:
                        kv_pred = qwen_json_qa(processor, model, img, prompt)
                    except Exception:
                        kv_pred = None
                rec = {
                    "dataset": "FUNSD",
                    "split": split,
                    "image": img_path,
                    "W": W,
                    "H": H,
                    "chunks": chunks,
                    "kv_pred": kv_pred,
                }
                fw.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run_chartqa(chart_root: Path, out_dir: Path, processor=None, model=None, use_ocr: bool = True, max_n: int = 200):
    qa_path = chart_root / "qa.jsonl"
    if not qa_path.exists():
        return
    out_jsonl = out_dir / "chartqa.jsonl"
    with open(qa_path, "r", encoding="utf-8") as fr, open(out_jsonl, "w", encoding="utf-8") as fw:
        for i, line in enumerate(fr):
            ex = json.loads(line)
            img_path = ex["image"]
            img = Image.open(img_path).convert("RGB")
            W, H = img.size
            chunks = ocr_chunks(img_path, lang="en") if use_ocr else []
            chart_pred = None
            if processor and model:
                question = ex.get("question", "")
                prompt = (
                    "You are a chart QA assistant. Answer the question strictly based on the chart image.\n"
                    f"Question: '{question}'\n"
                    "Return a STRICT JSON only:\n"
                    "{"
                    "  'answer': 'str|number',"
                    "  'evidence': [{'bbox':[x1,y1,x2,y2]}],"
                    "  'confidence': 0-1"
                    "}\n"
                    "Bounding boxes must be pixel coordinates referring to the input image. "
                    "If unsure, set 'answer' to 'unknown' and still provide the most relevant bbox."
                )
                try:
                    chart_pred = qwen_json_qa(processor, model, img, prompt)
                except Exception:
                    chart_pred = None
            rec = {
                "dataset": "ChartQA",
                "image": img_path,
                "question": ex.get("question", ""),
                "gold_answer": ex.get("answer", ""),
                "W": W,
                "H": H,
                "chunks": chunks,
                "pred": chart_pred,
            }
            fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if i + 1 >= max_n:
                break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--funsd_root", default="data/funsd")
    ap.add_argument("--chartqa_root", default="data/chartqa_mini")
    ap.add_argument("--out_dir", default="runs/extract")
    ap.add_argument("--model_dir", default="models/qwen")
    ap.add_argument("--no_mllm", action="store_true")
    ap.add_argument("--no_ocr", action="store_true")
    args = ap.parse_args()
    out = Path(args.out_dir)
    ensure_dirs(out)
    processor = None
    model = None
    if not args.no_mllm:
        processor, model = load_qwen(args.model_dir)
    run_funsd(Path(args.funsd_root), out, processor, model, use_ocr=not args.no_ocr)
    run_chartqa(Path(args.chartqa_root), out, processor, model, use_ocr=not args.no_ocr)


if __name__ == "__main__":
    main()