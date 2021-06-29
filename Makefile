.PHONY: init 

init:
	pip install -r requirements.txt && pip install -r requirements_dev.txt
	pre-commit install && pre-commit autoupdate