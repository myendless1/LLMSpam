nohup python visualize_spam.py --config_path bert_base --device 0 &
nohup python visualize_spam.py --config_path bert_base_freeze --device 1 &
nohup python visualize_spam.py --config_path bert_tiny --device 0 &
nohup python visualize_spam.py --config_path bert_tiny_freeze --device 1 &
