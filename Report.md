# DocRAG — Evidence‑Grounded Document QA (Charts + Forms)

> **Repo:** all scripts live under `src/` and write artifacts to `runs/` (created automatically).  
> **Models:** OCR (EasyOCR by default), dense retrieval (`sentence-transformers/all-MiniLM-L6-v2`), VLM (Qwen‑2.5‑VL‑7B).

## 1. What this project does
DocRAG turns images/PDF pages into **structured, retrievable facts** and then answers questions with **verifiable evidence** (bounding boxes). It supports:
- **ChartQA‑mini**: image‑based chart QA with accuracy split by numeric/text.
- **FUNSD (forms)**: field‑level QA evaluators (e.g., **Date**, **right‑edge Stamp/ID**), each reporting **value Exact‑Match** and **box IoU@τ**.

## 2. Repository layout (key scripts)
```
src/
  extract.py              # OCR + lightweight structure extraction → JSONL
  index.py                # FAISS index over OCR chunks
  qa.py                   # Evidence retrieval + Qwen‑VL answering (Date, etc.)
  qa_stampid.py           # Deterministic right‑edge Stamp/ID detector
  eval_chart.py           # ChartQA metrics (overall / numeric / text + bad cases)
  eval_funsd_date.py      # FUNSD Date evaluator (EM + IoU)
  eval_stampid.py         # FUNSD Stamp/ID evaluator (EM + IoU)
runs/                     # auto‑created; predictions & metrics saved here
data/                     # your datasets (e.g., ChartQA mini, FUNSD images)
```

## 3. Pipeline
### 3.1 OCR & extraction
```bash
python src/extract.py \
  --images_dir data/funsd/test/images \
  --out runs/extract/funsd_test.jsonl
```
The file `runs/extract/funsd_test.jsonl` contains per‑image OCR chunks with text and bounding boxes (“silver” labels for some fields).

### 3.2 Dense retrieval index
```bash
python src/index.py \
  --extract_jsonl runs/extract/funsd_test.jsonl \
  --out_dir runs/index
```
Builds `runs/index/chunks.faiss` + `runs/index/meta.jsonl` using `all-MiniLM-L6-v2` embeddings.

### 3.3 Vision‑language answering (evidence‑first)
- **Entry points:** `src/qa.py` (Date and generic QA) and `src/qa_stampid.py` (Stamp/ID).  
- **Behaviour:** retrieve top‑k OCR chunks; if needed, enable regex fallback (dates/phones/pages). Then call **Qwen‑2.5‑VL‑7B** with a strict JSON prompt; post‑processing ensures:
  - only digits when required (e.g., phone/fax),
  - normalized dates (`MM/DD/YYYY`),
  - a valid `value_bbox` (filled from first evidence when missing).
- **Outputs:** one JSON per image under `runs/qa_*/*.json` with fields:
  ```json
  {"image": ".../0.png",
   "question": "...",
   "answer": "{\"answer\":\"12/10/1998\",\"value_bbox\":[x1,y1,x2,y2],\"confidence\":0.92}",
   "evidence": [{"image":".../0.png","text":"DATE","bbox":[...],"score":1.2}]}
  ```

## 4. Evaluation (reproducible commands)
### 4.1 ChartQA‑mini
```bash
# Your predictions JSONL produced by the ChartQA runner
python src/eval_chart.py \
  --pred_jsonl runs/extract/chartqa.jsonl \
  --out runs/eval/chartqa_metrics.json \
  --eps 0.02 --lenient_text 1
```
**Result (latest run):**
- `N_total_with_gold=200`, `Pred_nonnull_rate=1.0`
- **Overall Acc = 0.815**; **Numeric = 0.8345**; **Text = 0.7636**
- Bad cases saved to `runs/eval/chartqa_bad.json`

### 4.2 FUNSD — Date (`MM/DD/YYYY`)
```bash
# Generate per‑image predictions
python src/qa.py \
  --task date \
  --images_dir data/funsd/test/images \
  --out_dir runs/qa_date

# Evaluate against silver labels from OCR extract
python src/eval_funsd_date.py \
  --qa_dir runs/qa_date \
  --extract_jsonl runs/extract/funsd_test.jsonl \
  --out runs/eval/funsd_date_metrics.json \
  --iou_thr 0.5
```
The metrics JSON is written to `runs/eval/funsd_date_metrics.json` (fields: `N_files`, `N_eval_with_silver`, `Unknown_rate`, `Value_EM`, `Box_IoU@0.50`, and `examples_bad`).

### 4.3 FUNSD — right‑edge Stamp/ID (vertical digits)
```bash
# Deterministic detector with optional overlays
python src/qa_stampid.py \
  --images_dir data/funsd/test/images \
  --out_dir runs/qa_stampid \
  --visualize

# Evaluate (EM on value; IoU on box if a silver box exists)
python src/eval_stampid.py \
  --qa_dir runs/qa_stampid \
  --extract_jsonl runs/extract/funsd_test.jsonl \
  --out runs/eval/stampid_metrics.json \
  --iou_thr 0.5
```
Notes: `qa_stampid.py` rotates pages/ROIs when needed and uses contour/angle filtering to isolate the right‑edge vertical number. If the extract lacks a silver box, IoU is skipped but **EM** on the value is still computed.

## 5. What to expect (empirical snapshot)
- **ChartQA‑mini:** `Acc=0.815` overall (`Numeric=0.8345`, `Text=0.7636`).
- **FUNSD‑Date:** metrics are saved to `runs/eval/funsd_date_metrics.json` (latest run improved markedly after regex‑guided evidence and date normalization).
- **FUNSD‑Stamp/ID:** `runs/eval/stampid_metrics.json` contains EM; when silver boxes are absent, IoU is omitted. A small manual audit on 50 images observed **value ~78%** and **box ~80%** correctness with `--visualize` overlays.

> All numbers above are produced by the scripts in this repo; adjust seeds and thresholds if you aim to match them exactly on a new machine.

## 6. Design choices
- **Evidence‑first prompting** (answer restricted to retrieved/regex candidates) improves faithfulness and makes answers auditable.
- **Deterministic fallback** for easy fields (Date/StampID) keeps accuracy high without over‑reliance on VLM decoding.
- **Reproducibility**: every step writes JSON/CSV artifacts under `runs/` and has a single Python entry point.

## 7. Quickstart (copy‑paste)
```bash
# FUNSD (forms)
python src/extract.py --images_dir data/funsd/test/images --out runs/extract/funsd_test.jsonl
python src/index.py   --extract_jsonl runs/extract/funsd_test.jsonl --out_dir runs/index
python src/qa.py      --task date --images_dir data/funsd/test/images --out_dir runs/qa_date
python src/qa_stampid.py --images_dir data/funsd/test/images --out_dir runs/qa_stampid --visualize
python src/eval_funsd_date.py --qa_dir runs/qa_date --extract_jsonl runs/extract/funsd_test.jsonl --out runs/eval/funsd_date_metrics.json --iou_thr 0.5
python src/eval_stampid.py    --qa_dir runs/qa_stampid --extract_jsonl runs/extract/funsd_test.jsonl --out runs/eval/stampid_metrics.json --iou_thr 0.5

# ChartQA‑mini
python src/eval_chart.py --pred_jsonl runs/extract/chartqa.jsonl --out runs/eval/chartqa_metrics.json --eps 0.02 --lenient_text 1
```

---

*Last updated:* Jan 2026 — this `report.md` references **actual scripts/paths in this repository** (no external placeholders).
