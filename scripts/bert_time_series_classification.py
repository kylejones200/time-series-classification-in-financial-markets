"""BERT for time series classification in financial markets."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments


class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, tokens, labels):
        self.tokens = tokens
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val.squeeze(0) for key, val in self.tokens[idx].items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def tokenize_time_series(tokenizer, series):
    series_str = " ".join(f"{v:.4f}" for v in series)
    return tokenizer(
        series_str,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
    )


def main_step_001() -> None:
    os.environ["WANDB_DISABLED"] = "true"
    np.random.seed(42)
    n_samples = 100
    n_timestamps = 50
    class_0 = np.random.normal(0, 1, (n_samples // 2, n_timestamps))
    class_1 = np.random.normal(2, 1, (n_samples // 2, n_timestamps))
    X = np.vstack((class_0, class_1))
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
    df = pd.DataFrame(X)
    df["label"] = y

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokens = [tokenize_time_series(tokenizer, row) for row in df.iloc[:, :-1].values]
    labels = df["label"].values
    train_tokens, test_tokens, train_labels, test_labels = train_test_split(
        tokens, labels, test_size=0.2, random_state=42
    )
    train_dataset = TimeSeriesDataset(train_tokens, train_labels)
    test_dataset = TimeSeriesDataset(test_tokens, test_labels)

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    training_args = TrainingArguments(
        output_dir="./outputs/bert_checkpoints",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        save_steps=50,
        save_total_limit=1,
        logging_steps=20,
        report_to="none",
        eval_strategy="no",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    trainer.train()
    predictions = trainer.predict(test_dataset)
    predicted_labels = np.argmax(predictions.predictions, axis=1)
    accuracy = accuracy_score(test_labels, predicted_labels)
    print(f"Test accuracy: {accuracy:.2%}")


def main() -> None:
    main_step_001()


if __name__ == "__main__":
    main()
