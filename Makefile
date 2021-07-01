.PHONY: init clean

init:
	pip install -r requirements_min.txt && pip install -r requirements_dev.txt
	pip freeze > requirements.txt
	pre-commit install && pre-commit autoupdate

clean:
	rm -f ./data/interim/*

./data/interim/triples_filtered_by_predicate.csv: ./data/raw/hc_dump_latest.csv ./config/predicate_filter.csv
	python src/cli/filter_data_by_predicate.py -i ./data/raw/hc_dump_latest.csv -o ./data/interim/triples_filtered_by_predicate.csv -p ./config/predicate_filter.csv