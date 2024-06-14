from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
from torch.utils.data import Dataset
from tqdm import tqdm, trange
import numpy as np
from process_data import remove_header

llm_path = "LLM/chatglm3-6b"

tokenizer = AutoTokenizer.from_pretrained(llm_path,trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(llm_path,trust_remote_code=True)
model = model.to("cuda")

spam_email = """
Received: from 59-117-56-198.dynamic.hinet.net ([59.117.56.198]) by groucho.cs.psu.edu
	with SMTP id <2589>; Sun, 11 Apr 1993 05:50:39 -0400
Received: from bug.downstate.pt  by exclusionary.agrimony.pt with Microsoft SMTPSVC(5.0.4465.2338);
	Sun, 11 Apr 1993 10:48:37 +0400
Date: Sun, 11 Apr 1993 11:47:37 +0500
From: "Jody Gee" <cdysd@kodiakcomputer.com>
To: marisol@groucho.cs.psu.edu
Cc: casandra@groucho.cs.psu.edu, aurelia@groucho.cs.psu.edu, herman@groucho.cs.psu.edu, milagros@groucho.cs.psu.edu, kurt@groucho.cs.psu.edu, henrietta@groucho.cs.psu.edu, jim@groucho.cs.psu.edu, eleanor@groucho.cs.psu.edu
Subject: Burn body fat
Message-ID: <cdysd@kodiakcomputer.com.pt>
MIME-Version: 1.0
Content-Type: text/plain; charset=ISO-8859-1; format=flowed
Content-Transfer-Encoding: 8bit
From NBC Today Show:
It's the look everyone wants � a body to diet for.
They're on the beaches, in magazines and all over Hollywood. How far will we go to get one?
How about thousands of miles and deep into a distant culture?
South Africa�s Kalahari Desert is home to what could be the answer to an appetite.
It's a cactus called hoodia. �You strip off the skin, you strip off the spines,
and then you consume it,� says weight loss expert Madelyn Fernstrom.
It`s a revolution! Read more....
http://051.thingswithdiets.com
picture you borne me, hardtack erato . plagiarism you pallet me, inculcate shoo . basilar you extravaganza me, brighten drove . emilio you embrittle me, eileen writhe . 
blackboard you demolition me, hesitant inhibit silhouette tribesmen . baden you doyle me, andes guanine carbonic . spud you eloise me, redstart fore evaporate . penitent you solvent me, alveolus formatted othello . 
http://051.thingswithdiets.com/rm/
"""

ham_email = """
Received: from psuvax1.cs.psu.edu ([130.203.2.4]) by groucho.cs.psu.edu with SMTP id <2579>; Sat, 10 Apr 1993 21:11:57 -0400
Received: from groucho.cs.psu.edu ([130.203.2.10]) by psuvax1.cs.psu.edu with SMTP id <292444>; Sat, 10 Apr 1993 21:12:21 -0400
Received: from localhost by groucho.cs.psu.edu with SMTP id <2579>; Sat, 10 Apr 1993 21:10:56 -0400
To:	9fans <plan9-fans@cs.psu.edu>
Subject: who wants to start?
Date:	Sat, 10 Apr 1993 21:10:19 -0400
From:	Scott Schwartz <schwartz@groucho.cs.psu.edu>
Message-Id: <93Apr10.211056edt.2579@groucho.cs.psu.edu>

It's quiet.  Too quiet.  Well, how about a straw poll then.
How many have plan9 running yet?
"""

new_email ="""
"""



def pred_label(new_email, model, tokenizer):
    messages = []
    messages.append({"role": "system", "content": "You are an experienced spam email classifier. You are given a spam email and you need to classify it as [[Spam]] or [[Ham]]."})
    messages.append({"role": "user", "content": """
    Here are two examples of emails, followed by their labels:
    ## Email
    {email1}
    ## Label, please choose from [[Spam]] or [[Ham]]
    [[Spam]].
    ## Email
    {email2}
    ## Label, please choose from [[Spam]] or [[Ham]]
    [[Ham]].
    Now, you are given a new email. Based on the content, please classify it as either spam or not spam. Use the same format as above, and ensure either [[Spam]] or [[Ham]] is included in your response. Use double brackets to indicate the label.
    Do not introduce Any bias in the classification.
    The email is:
    ## Email
    {email3}
    ## Label, please choose from [[Spam]] or [[Ham]]
    """.format(email1=spam_email, email2=ham_email, email3=new_email)})
    model_inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt")
    model_inputs = model_inputs.to("cuda")

    generate_kwargs = {
        "input_ids": model_inputs,
        "max_new_tokens": 256,
        "temperature": 0.0,
        "do_sample": False
    }
    output = model.generate(**generate_kwargs)
    decoded = tokenizer.batch_decode(output[:, model_inputs.size(1):], skip_special_tokens=True)
    with open("infer_llm.log", "a") as f:
        f.write(f"Email: {new_email}\n")
        f.write(f"Response: {decoded}\n")
    try:
        label = re.search(r"\[\[(\w+)\]\]", decoded[0]).group(1)
    except:
        if "ham" in decoded[0].lower():
            label = "Ham"
        elif "spam" in decoded[0].lower():
            label = "Spam"
    label = 0 if label == "Ham" else 1 if label == "Spam" else -1
    if label == -1:
        print(f"Error: {new_email}\n", f"Response: {decoded}\n")
    return decoded, label


class TempTrec06pDataset(Dataset):
    def __init__(self, root_path=None, index_path=None, split="train", train_size=0.9):
        index_file = open(f"{root_path}/{index_path}", "r")
        self.labels = []
        self.file_contents = []
        lines = index_file.readlines()
        np.random.shuffle(lines)
        if split == "train":
            lines = lines[:int(train_size * len(lines))]
        else:
            lines = lines[int(train_size * len(lines)):]
        for index in tqdm(lines):
            typ, path = tuple(index.split(" "))
            label = 0 if typ == "ham" else 1
            self.labels.append(label)
            path = path.replace("../", "")
            file_path = f"{root_path}/{path}".strip()
            file_content = open(file_path, "r", encoding="iso-8859-1").read()
            self.file_contents.append(file_content)

    def __len__(self):
        return len(self.file_contents)

    def __getitem__(self, idx):
        return self.file_contents[idx], self.labels[idx]
    
    def split_train_test(self, train_size=0.8):
        train_size = int(train_size * len(self.data))
        return TempTrec06pDataset(data=self.data[:train_size]), TempTrec06pDataset(data=self.data[train_size:])

trec_ds = TempTrec06pDataset("trec06p", "full/index", split="test", train_size=0.9)


gt_labels = []
pred_labels = []
for i in trange(len(trec_ds)):
# for i in trange(100):
    email, label = trec_ds[i]
    gt_labels.append(label)
    email = remove_header(email)
    decoded, predicted_label = pred_label(email[:2048], model, tokenizer)
    pred_labels.append(predicted_label)

TP = sum([1 for gt, pred in zip(gt_labels, pred_labels) if gt == 1 and pred == 1])
FP = sum([1 for gt, pred in zip(gt_labels, pred_labels) if gt == 0 and pred == 1])
TN = sum([1 for gt, pred in zip(gt_labels, pred_labels) if gt == 0 and pred == 0])
FN = sum([1 for gt, pred in zip(gt_labels, pred_labels) if gt == 1 and pred == 0])

accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * precision * recall / (precision + recall)
print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")
with open("infer_llm.result", "w") as f:
    f.write(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}\n")

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
# remove the examples where label is -1
gt_labels = [gt for gt, pred in zip(gt_labels, pred_labels) if pred != -1]
pred_labels = [pred for pred in pred_labels if pred != -1]
data = {
    "y_Actual": gt_labels,
    "y_Predicted": pred_labels
}
df = pd.DataFrame(data, columns=["y_Actual", "y_Predicted"])
confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])

sns.heatmap(confusion_matrix, annot=True)
plt.savefig(f"spam_confusion_llm.png")
plt.close()
