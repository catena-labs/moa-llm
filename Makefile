.PHONY : run-checks
run-checks :
	isort --check .
	black --check .
	ruff check .

.PHONY : build
build :
	rm -rf *.egg-info/
	python -m build
