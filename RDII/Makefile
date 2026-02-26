.PHONY: install clean test docs

install:
	pip install -e .

test:
	pytest tests/

clean:
	rm -rf __pycache__ .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +

docs:
	cd docs && make html

docs-clean:
	cd docs && make clean