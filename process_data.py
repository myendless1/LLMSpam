from torch.utils.data import Dataset
from tqdm import tqdm
import torch
import numpy as np
from transformers import BertTokenizer
import h5py

head_words = [
    "Received:", "To:", 
    "From:", "Subject:", 
    "MIME-Version:", "Content-Type:", 
    "Reply-To:", "Content-Transfer-Encoding:",
    "Importance:", "Message-Id:",
    "Date:", "X-Bogosity:",
    "X-Spam-Score:", "X-Scanned-By:",
    ]

def remove_header(email):
    valid_lines = []
    for line in email.split("\n"):
        has_head = False
        for head in head_words:
            if head in line:
                valid_lines = []
                has_head = True
                break
        if not has_head:
            valid_lines.append(line)
    return "\n".join(valid_lines)

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
        for index in tqdm(lines[:num_samples]):
            typ, path = tuple(index.split(" "))
            label = 0 if typ == "ham" else 1
            labels.append(label)
            path = path.replace("../", "")
            file_path = f"{root_path}/{path}".strip()
            file_content = open(file_path, "r", encoding="iso-8859-1").read()
            file_contents.append(remove_header(file_content))
        encoded = tokenizer.batch_encode_plus(file_contents, max_length=max_seq_length, pad_to_max_length=True)
        for i in range(len(encoded["input_ids"])):
            self.data.append((torch.tensor(encoded["input_ids"][i]), torch.tensor(encoded["attention_mask"][i]),
                                torch.tensor(labels[i])))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def split_train_test(self, train_size=0.8):
        train_size = int(train_size * len(self.data))
        return TempTrec06pDataset(data=self.data[:train_size]), TempTrec06pDataset(data=self.data[train_size:])
    
if __name__ == '__main__':
    data_path = "trec06p"
    full_index_path = "full/index"
    bert_path = "bert-tiny"
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    ds = TempTrec06pDataset(data_path, full_index_path, tokenizer=tokenizer, num_samples=-1)
    train_ds, test_ds = ds.split_train_test(train_size=0.9)
    file = h5py.File("data/trec06p_train.h5", "w")
    file.create_dataset("input_ids", data=[x[0] for x in train_ds])
    file.create_dataset("attention_mask", data=[x[1] for x in train_ds])
    file.create_dataset("labels", data=[x[2] for x in train_ds])
    file.close()
    file = h5py.File("data/trec06p_test.h5", "w")
    file.create_dataset("input_ids", data=[x[0] for x in test_ds])
    file.create_dataset("attention_mask", data=[x[1] for x in test_ds])
    file.create_dataset("labels", data=[x[2] for x in test_ds])
    file.close()
