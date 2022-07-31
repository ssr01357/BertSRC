import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, TrainingArguments, Trainer

from model import BertSrcClassifier
from utils import SrcDataset, compute_metrics, CLASSES


test = pd.read_csv("data/test.csv", encoding='utf-8')

tokenizer = BertTokenizer.from_pretrained(
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
)
model = BertSrcClassifier.from_pretrained(
    "./checkpoints/best_model",
    num_labels=len(CLASSES),
    mask_token_id=tokenizer.mask_token_id,
)

test_x1 = test.apply(lambda x: x["14"].replace(x["3"], "[MASK]"), axis=1).tolist()
test_x2 = test.apply(lambda x: x["14"].replace(x["8"], "[MASK]"), axis=1).tolist()

label_encoder = LabelEncoder()
test_y = torch.tensor(label_encoder.fit_transform(test["16"]), dtype=torch.long)

tokenized_test = tokenizer(
    test_x1, test_x2, return_tensors="pt", padding=True, truncation=True
)

test_dataset = SrcDataset(tokenized_test, test_y)

training_args = TrainingArguments(
    output_dir="./checkpoints",
    per_device_eval_batch_size=64,
)

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.evaluate()
