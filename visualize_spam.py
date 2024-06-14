import torch
import argparse
from configs import get_config
from torch.utils.data import DataLoader
from train import BertPalmClassifier, Trec06pDataset

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
    model._load_from_pth(f"ckpts/{config.ckpt_name}.pth")
    model.eval()

    test_ds = Trec06pDataset("data/trec06p_test.h5")
    test_dataloader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=True)

    feacture_vectors = []
    feature_logits = []
    gt_labels = []
    pred_labels = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in test_dataloader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            with torch.no_grad():
                gt_labels.extend(labels.cpu().numpy().tolist())
                output, logits = model(input_ids, attention_mask, return_features=True)
                for i in range(len(output)):
                    feacture_vectors.append(output[i].cpu().numpy())
                    feature_logits.append(logits[i].cpu().numpy())
                pred_labels.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())


    # visualize these vectors in a 2D plot
    import matplotlib.pyplot as plt
    import numpy as np

    feacture_vectors = np.array(feacture_vectors)
    feature_logits = np.array(feature_logits)
    print(feacture_vectors.shape)
    print(feature_logits.shape)
    plt.figure(figsize=(10, 10))
    plt.scatter(feature_logits[:, 0], feature_logits[:, 1], c=gt_labels, cmap="coolwarm", s=1)
    plt.colorbar()
    plt.savefig(f"spam_feature_{args.config_path}.png")
    plt.close()
    feature_logits = torch.softmax(torch.tensor(feature_logits), dim=1).numpy()
    plt.figure(figsize=(10, 10))
    plt.scatter(feature_logits[:, 0], feature_logits[:, 1], c=gt_labels, cmap="coolwarm", s=1)
    plt.colorbar()
    plt.savefig(f"spam_feature_softmax_{args.config_path}.png")
    plt.close()
    TP = sum([1 for i in range(len(gt_labels)) if gt_labels[i] == 1 and pred_labels[i] == 1])
    TN = sum([1 for i in range(len(gt_labels)) if gt_labels[i] == 0 and pred_labels[i] == 0])
    FP = sum([1 for i in range(len(gt_labels)) if gt_labels[i] == 0 and pred_labels[i] == 1])
    FN = sum([1 for i in range(len(gt_labels)) if gt_labels[i] == 1 and pred_labels[i] == 0])
    # draw confusion matrix
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    data = {
        "y_Actual": gt_labels,
        "y_Predicted": pred_labels
    }
    df = pd.DataFrame(data, columns=["y_Actual", "y_Predicted"])
    confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])

    sns.heatmap(confusion_matrix, annot=True)
    plt.savefig(f"spam_confusion_{args.config_path}.png")
    plt.close()
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    print(f"{config.ckpt_name}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")


