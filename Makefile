.PHONY: build commit quality style test

check_dirs := src tests setup.py

build:
	pip install build && python -m build

commit:
	pre-commit install
	pre-commit run --all-files

quality:
	python -m compileall -q src tests
	CUDA_VISIBLE_DEVICES= WANDB_DISABLED=true pytest -q tests/

style:
	ruff check $(check_dirs) --fix
	ruff format $(check_dirs)

test:
	CUDA_VISIBLE_DEVICES= WANDB_DISABLED=true pytest -vv tests/
