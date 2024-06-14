from torch.utils.data import Dataset
from tqdm import tqdm
import torch
import numpy as np
from transformers import BertTokenizer
import h5py

class TempTrec06pDataset(Dataset):
    def __init__(self, root_path=None, index_path=None, max_seq_length=512, tokenizer=None, num_samples=-1, data=None):
        if data is not None:
            self.data = data
            return
        else:
            assert root_path is not None and index_path is not None and tokenizer is not None, "Need to provide root_path, index_path and tokenizer"
        index_file = open(f"{root_path}/{index_path}", "r")
        self.max_seq_length = max_seq_length
        self.data = []
        self.tokenizer = tokenizer
        labels = []
        file_contents = []
        lines = index_file.readlines()
        np.random.shuffle(lines)
        self.lengths = []
        for index in tqdm(lines[:num_samples]):
            typ, path = tuple(index.split(" "))
            label = 0 if typ == "ham" else 1
            labels.append(label)
            path = path.replace("../", "")
            file_path = f"{root_path}/{path}".strip()
            file_content = open(file_path, "r", encoding="iso-8859-1").read()
            file_contents.append(file_content.replace("\n", "").replace("\r", ""))
            tokens = tokenizer.tokenize(file_content)
            length = len(tokens)
            self.lengths.append(length)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return None
    
    def get_lengths(self):
        return self.lengths
    
if __name__ == '__main__':
    data_path = "trec06p"
    full_index_path = "full/index"
    bert_path = "bert-tiny"
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    ds = TempTrec06pDataset(data_path, full_index_path, tokenizer=tokenizer, num_samples=-1)
    lengths = ds.get_lengths()
    max_length = max(lengths)
    min_length = min(lengths)
    avg_length = sum(lengths) / len(lengths)
    print(f"Max length: {max_length}")
    print(f"Min length: {min_length}")
    print(f"Avg length: {avg_length}")
    # train_ds, test_ds = ds.split_train_test(train_size=0.9)
    # file = h5py.File("data/trec06p_train.h5", "w")
    # file.create_dataset("input_ids", data=[x[0] for x in train_ds])
    # file.create_dataset("attention_mask", data=[x[1] for x in train_ds])
    # file.create_dataset("labels", data=[x[2] for x in train_ds])
    # file.close()
    # file = h5py.File("data/trec06p_test.h5", "w")
    # file.create_dataset("input_ids", data=[x[0] for x in test_ds])
    # file.create_dataset("attention_mask", data=[x[1] for x in test_ds])
    # file.create_dataset("labels", data=[x[2] for x in test_ds])
    # file.close()

