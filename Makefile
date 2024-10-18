.PHONY: style format


style:
	python -m black --line-length 119 .
	python -m black --line-length 119 src
	python -m isort .
	python -m isort src
	ruff check --fix .
	ruff check --fix src


quality:
	python -m black --check --line-length 119 .
	python -m black --check --line-length 119 src
	python -m isort --check-only .
	python -m isort --check-only src
	ruff check .
	ruff check src


test:
	python -m pytest tests