nohup python train.py --config_path bert_base --device 1 > log_bert_base.txt &
nohup python train.py --config_path bert_base_freeze --device 1 > log_bert_base_freeze.txt &
nohup python train.py --config_path bert_tiny --device 1 > log_bert_tiny.txt &
nohup python train.py --config_path bert_tiny_freeze --device 1 > log_bert_tiny_freeze.txt &
