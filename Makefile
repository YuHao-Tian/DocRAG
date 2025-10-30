VENV=.venv
PY=$(VENV)/bin/python
PIP=$(VENV)/bin/pip

init:
	python -m venv $(VENV)
	. $(VENV)/bin/activate; $(PIP) install -r requirements.txt

data: $(VENV)
	$(PY) scripts/get_funsd.py
	$(PY) scripts/get_chartqa_mini.py

extract:
	mkdir -p runs/extract
	$(PY) src/extract.py --funsd_root data/funsd --chartqa_root data/chartqa_mini --out_dir runs/extract --model_dir models/qwen

index:
	mkdir -p runs/index
	$(PY) src/index.py --extract_dir runs/extract --db_dir runs/index

qa:
	mkdir -p runs/qa
	$(PY) src/qa.py --db_dir runs/index --extract_dir runs/extract --model_dir models/qwen --image data/chartqa_mini/images/sample_0.png --q "What is the value?" --out runs/qa/answer.json

viz:
	mkdir -p runs/figs
	$(PY) src/viz.py --qa_json runs/qa/answer.json --out runs/figs/overlay.png

eval:
	mkdir -p runs/eval
	$(PY) scripts/eval_chart.py --pred_jsonl runs/extract/chartqa.jsonl --out runs/eval/chartqa_metrics.json
	$(PY) scripts/eval_funsd.py --extract_jsonl runs/extract/funsd_test.jsonl --qa_dir runs/qa --out runs/eval/funsd_metrics.json
