.PHONY: install test lint smoke train eval demo

install:
	python -m pip install -r requirements.txt

test:
	python -m pytest

lint:
	ruff check .

smoke:
	python scripts/smoke_test.py --epochs 3 --use-synthetic

train:
	python scripts/train.py --config configs/default.yaml

eval:
	python scripts/evaluate.py --checkpoint $(CHECKPOINT) --data-root $(DATA_ROOT)

demo:
	streamlit run app.py
