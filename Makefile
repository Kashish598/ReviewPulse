.PHONY: setup data fixture lint test api

setup:
	pip install -r requirements.txt

data:
	python -m src.prepare_data

fixture:
	python -m src.prepare_data --fixture

lint:
	ruff check .

test:
	pytest -v

api:
	uvicorn src.app:app --reload
