set -e
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python scripts/get_funsd.py
python scripts/get_chartqa_mini.py
mkdir -p runs/extract runs/index runs/qa runs/figs runs/eval
python src/extract.py --funsd_root data/funsd --chartqa_root data/chartqa_mini --out_dir runs/extract --model_dir models/qwen
python src/index.py --extract_dir runs/extract --db_dir runs/index
python src/qa.py --db_dir runs/index --extract_dir runs/extract --model_dir models/qwen --image data/chartqa_mini/images/sample_0.png --q "What is the value?" --out runs/qa/answer.json
python src/viz.py --qa_json runs/qa/answer.json --out runs/figs/overlay.png
python scripts/eval_chart.py --pred_jsonl runs/extract/chartqa.jsonl --out runs/eval/chartqa_metrics.json
python scripts/eval_funsd.py --extract_jsonl runs/extract/funsd_test.jsonl --qa_dir runs/qa --out runs/eval/funsd_metrics.json
