# DocRAG Research Repository

This repository contains an end‑to‑end document and chart question answering pipeline.  It converts PDF pages or images into structured JSON with bounding boxes, builds a vector index for retrieval, answers questions using a vision–language model, and draws evidence overlays for visualization.

## Directory layout

* `src/` – core Python modules for extraction, indexing, question answering and visualization.
* `scripts/` – utilities for downloading datasets and evaluating predictions on ChartQA and FUNSD.
* `data/` – place datasets such as FUNSD and ChartQA here (automatically created by the download scripts).
* `runs/` – output directory for intermediate results: extracted chunks, vector indices, QA outputs and evaluation metrics.
* `models/` – place your Qwen2.5 vision–language model checkpoint here.

## Installation

Install the dependencies from the provided `requirements.txt` file.  A typical setup on a recent Python distribution looks like this:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

You will also need a local copy of the Qwen2.5 vision–language checkpoint (e.g. `Qwen2.5‑VL‑7B‑Instruct_weights`).  Place it under `models/qwen` or specify a custom `--model_dir` when running the scripts below.

## Datasets

Two datasets are used in the experiments:

1. **FUNSD** – a scanned form dataset.  Run `scripts/get_funsd.py` to download and prepare the dataset:

   ```bash
   python scripts/get_funsd.py
   ```

   This script downloads the dataset from HuggingFace, saves images under `data/funsd/{train,test}/images` and writes the annotations into `labels.jsonl` files.

2. **ChartQA mini** – a small subset of ChartQA examples.  Run `scripts/get_chartqa_mini.py` to download and prepare the dataset:

   ```bash
   python scripts/get_chartqa_mini.py
   ```

   Images are stored under `data/chartqa_mini/images` and questions/answers in `qa.jsonl`.

## Extracting text and optional key‑value pairs

Use `src/extract.py` to build OCR and (optionally) model‑based key‑value extractions from FUNSD pages and ChartQA charts.  Extraction outputs are written to JSONL files under `runs/extract`.

```bash
python src/extract.py \
  --funsd_root data/funsd \
  --chartqa_root data/chartqa_mini \
  --out_dir runs/extract \
  --model_dir models/qwen
```

The script detects chunks from provided annotations and EasyOCR, and can call the Qwen model to propose generic key–value pairs (`--no_mllm` disables the model and `--no_ocr` disables the OCR fallback).

## Building a dense retrieval index

After extraction you can build a FAISS index over the text chunks for retrieval.  Use `src/index.py`:

```bash
python src/index.py --extract_dir runs/extract --db_dir runs/index
```

This writes a vector index (`chunks.faiss`) and a metadata file (`meta.jsonl`) into `runs/index`.

## Question answering

To answer questions given an image and a query, run `src/qa.py`.  It retrieves top‑k evidence chunks from the index, optionally falls back to regex heuristics when no evidence is found, and then calls Qwen2.5 to produce an answer and bounding box.

Example:

```bash
python src/qa.py \
  --db_dir runs/index \
  --extract_dir runs/extract \
  --model_dir models/qwen \
  --image path/to/image.png \
  --q "What is the date?" \
  --out runs/qa/answer.json
```

The output JSON includes the image path, question, model answer (as a JSON string), and the evidence chunks used for retrieval.

## Visualization

You can draw bounding boxes showing both retrieval evidence and the model’s predicted value bounding box using `src/viz.py`:

```bash
python src/viz.py --qa_json runs/qa/answer.json --out runs/figs/overlay.png
```

## Evaluation

The evaluation scripts compute accuracies on ChartQA and FUNSD without relying on any Chinese comments or annotations.

### ChartQA

```bash
python scripts/eval_chart.py \
  --pred_jsonl runs/extract/chartqa.jsonl \
  --out runs/eval/chartqa_metrics.json
```

This script reads the predictions stored in `pred` fields of the extraction output and compares them against the gold answers.

### FUNSD key fields

```bash
python scripts/eval_funsd.py \
  --extract_jsonl runs/extract/funsd_test.jsonl \
  --qa_dir runs/qa \
  --out runs/eval/funsd_metrics.json
```

It measures value and bounding‑box accuracy for dates, phone numbers, fax numbers and page counts.
