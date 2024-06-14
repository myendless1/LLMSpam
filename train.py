from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import torch
import argparse
from configs import get_config
from tensorboardX import SummaryWriter

class Trec06pDataset(Dataset):
    def __init__(self, data_path):
        file = h5py.File(data_path, "r")
        self.input_ids = file["input_ids"]
        self.attention_mask = file["attention_mask"]
        self.labels = file["labels"]

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx], self.labels[idx]

class BertPalmClassifier(nn.Module):
    def __init__(self, bert_path, num_classes, freeze_bert=False):
        super(BertPalmClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        # freeze all but the classifier
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
            for param in self.classifier.parameters():
                param.requires_grad = True

    def forward(self, input_ids, attention_mask, return_features=False):
        output = self.bert(input_ids, attention_mask)
        pooled_output = output.pooler_output
        last_hidden_state = output.last_hidden_state
        if return_features:
            return pooled_output, self.classifier(pooled_output)
        return self.classifier(pooled_output)
    
    def _load_from_pth(self, path):
        self.load_state_dict(torch.load(path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="bert_base")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    config_path = f"configs/{args.config_path}.yaml"
    config = get_config(config_path)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    model = BertPalmClassifier(config.bert_path, 2, config.freeze_bert).to(device)

    train_ds = Trec06pDataset("data/trec06p_train.h5")
    test_ds = Trec06pDataset("data/trec06p_test.h5")
    train_dataloader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=True)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()
    summary_writer = SummaryWriter(log_dir=f"{config.log_dir}/{config.exp_name}")
    overall_steps = 0
    best_f1 = 0
    for epoch in range(10):
        for i, batch in tqdm(enumerate(train_dataloader)):
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(input_ids, attention_mask)
            loss = criterion(output, labels)
            loss.backward()
            summary_writer.add_scalar("loss", loss, i)
            optimizer.step()
            overall_steps += 1
            if overall_steps % config.val_interval == 0:
                model.eval()
                TP = 0
                TN = 0
                FP = 0
                FN = 0
                print("Evaluating...")
                for i, batch in tqdm(enumerate(test_dataloader)):
                    input_ids, attention_mask, labels = batch
                    input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
                    output = model(input_ids, attention_mask)
                    predicted = torch.argmax(output, dim=1)
                    for j in range(len(predicted)):
                        if predicted[j] == labels[j]:
                            if predicted[j] == 1:
                                TP += 1
                            else:
                                TN += 1
                        else:
                            if predicted[j] == 1:
                                FP += 1
                            else:
                                FN += 1
                accuracy = (TP + TN) / (TP + TN + FP + FN)
                precision = TP / (TP + FP)
                recall = TP / (TP + FN)
                f1 = 2 * precision * recall / (precision + recall)
                summary_writer.add_scalar("accuracy", accuracy, overall_steps)
                summary_writer.add_scalar("precision", precision, overall_steps)
                summary_writer.add_scalar("recall", recall, overall_steps)
                summary_writer.add_scalar("f1", f1, overall_steps)
                if f1 > best_f1:
                    best_f1 = f1
                    torch.save(model.state_dict(), f"ckpts/{config.ckpt_name}.pth")
                model.train()