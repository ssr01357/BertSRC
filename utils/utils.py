import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)

CLASSES = [
    "Directed Link",
    "Negative Cause",
    "Negative Decrease",
    "Negative Increase",
    "Positive Cause",
    "Positive Decrease",
    "Positive Increase",
    "Undirected Link",
]

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro"
    )
    acc = accuracy_score(labels, preds)
    print(classification_report(labels, preds, digits=3, target_names=CLASSES))
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


class SrcDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
