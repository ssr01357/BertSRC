import pandas as pd
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, TrainingArguments, Trainer

from model import BertSrcClassifier
from utils import preprocess_data, compute_metrics, CLASSES

if __name__ == "__main__":
    train = pd.read_csv("data/train.csv", encoding="utf-8")
    validation = pd.read_csv("data/validation.csv", encoding="utf-8")

    tokenizer = BertTokenizer.from_pretrained(
        "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    )
    model = BertSrcClassifier.from_pretrained(
        "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        num_labels=len(CLASSES),
        mask_token_id=tokenizer.mask_token_id,
    )

    train_dataset = preprocess_data(train, tokenizer, x_col="14", y_col="16", e1_col="3", e2_col="8")
    validation_dataset = preprocess_data(train, tokenizer, x_col="14", y_col="16", e1_col="3", e2_col="8")

    training_args = TrainingArguments(
        output_dir="./checkpoints",
        logging_dir="./logs",
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
        eval_dataset=validation_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model("./checkpoints/best_model")
