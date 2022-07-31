import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, TrainingArguments, Trainer

from model import BertSrcClassifier
from utils import SrcDataset, compute_metrics, CLASSES


train = pd.read_csv("data/train.csv", encoding="utf-8")
validate = pd.read_csv("data/validate.csv", encoding="utf-8")

tokenizer = BertTokenizer.from_pretrained(
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
)
model = BertSrcClassifier.from_pretrained(
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    num_labels=len(CLASSES),
    mask_token_id=tokenizer.mask_token_id,
)

train_x1 = train.apply(lambda x: x["14"].replace(x["3"], "[MASK]"), axis=1).tolist()
train_x2 = train.apply(lambda x: x["14"].replace(x["8"], "[MASK]"), axis=1).tolist()

validate_x1 = validate.apply(
    lambda x: x["14"].replace(x["3"], "[MASK]"), axis=1
).tolist()
validate_x2 = validate.apply(
    lambda x: x["14"].replace(x["8"], "[MASK]"), axis=1
).tolist()

label_encoder = LabelEncoder()
train_y = torch.tensor(label_encoder.fit_transform(train["16"]), dtype=torch.long)
validate_y = torch.tensor(label_encoder.transform(validate["16"]), dtype=torch.long)

tokenized_train = tokenizer(
    train_x1, train_x2, return_tensors="pt", padding=True, truncation=True
)
tokenized_validate = tokenizer(
    validate_x1, validate_x2, return_tensors="pt", padding=True, truncation=True
)

max_sequence_length = max(
    len(tokenized_train["input_ids"][0]), len(tokenized_validate["input_ids"][0])
)
tokenized_train = tokenizer(
    train_x1,
    train_x2,
    return_tensors="pt",
    padding="max_length",
    max_length=max_sequence_length,
)
tokenized_validate = tokenizer(
    validate_x1,
    validate_x2,
    return_tensors="pt",
    padding="max_length",
    max_length=max_sequence_length,
)

train_dataset = SrcDataset(tokenized_train, train_y)
validate_dataset = SrcDataset(tokenized_validate, validate_y)

training_args = TrainingArguments(
    output_dir="./checkpoints",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    num_train_epochs=10,
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_ratio=0.1,
    weight_decay=0.01,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    max_grad_norm=1,
    lr_scheduler_type="linear",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    seed=42,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validate_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("./checkpoints/best_model")
