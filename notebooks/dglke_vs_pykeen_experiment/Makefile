.PHONY: pykeen dglke

pykeen:
	python run_pykeen.py > pykeen_log.txt
	
dglke:
	DGLBACKEND=pytorch dglke_train --model_name RotatE -de --dataset heritageconnector --data_path '../../data/interim/train_test_split/' --delimiter '	' --data_files train.csv val.csv test.csv --format raw_udd_hrt --max_step 134075 --test --valid --batch_size 2000 --neg_sample_size 50 --hidden_dim 400 --lr 0.02 --neg_sample_size_eval 50 --log_interval 10000 --batch_size_eval 8 --save_path dglke_results/ --gpu 0 --no_eval_filter --mix_cpu_gpu --async_update > dglke_log.txt
