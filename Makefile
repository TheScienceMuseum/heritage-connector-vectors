.PHONY: init 

init:
	pip install -r requirements_min.txt && pip install -r requirements_dev.txt
	pip freeze > requirements.txt
	pre-commit install && pre-commit autoupdate