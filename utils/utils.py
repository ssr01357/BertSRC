import torch
from sklearn.preprocessing import LabelEncoder
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
    """compute metrics for huggingface transformers trainer
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro"
    )
    acc = accuracy_score(labels, preds)
    print(classification_report(labels, preds, digits=3, target_names=CLASSES))
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

def preprocess_data(df, tokenizer, x_col, y_col, e1_col, e2_col):
    x1 = df.apply(lambda x: x[x_col].replace(x[e1_col], "[MASK]"), axis=1).tolist()
    x2 = df.apply(lambda x: x[x_col].replace(x[e2_col], "[MASK]"), axis=1).tolist()

    label_encoder = LabelEncoder()
    y = torch.tensor(label_encoder.fit_transform(df[y_col]), dtype=torch.long)

    tokenized_x = tokenizer(x1, x2, return_tensors="pt", padding='max_length', truncation=True, max_length=512)

    dataset = SrcDataset(tokenized_x, y)
    return dataset

class SrcDataset(torch.utils.data.Dataset):
    """Dataset class for BertSRC
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)
