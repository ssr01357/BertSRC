import pandas as pd
from transformers import BertTokenizer, TrainingArguments, Trainer

from model import BertSrcClassifier
from utils import preprocess_data, compute_metrics, CLASSES


if __name__ == "__main__":
    test = pd.read_csv("data/test.csv", encoding="utf-8")

    tokenizer = BertTokenizer.from_pretrained(
        "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    )
    model = BertSrcClassifier.from_pretrained(
        "./checkpoints/best_model",
        num_labels=len(CLASSES),
        mask_token_id=tokenizer.mask_token_id,
    )

    test_dataset = preprocess_data(test, tokenizer, x_col="14", y_col="16", e1_col="3", e2_col="8")

    training_args = TrainingArguments(
        output_dir="./checkpoints",
        logging_dir="./logs",
        per_device_eval_batch_size=64,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.evaluate()
