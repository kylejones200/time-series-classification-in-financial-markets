"""Generated from Jupyter notebook: BERT for Time Series Classification

Magics and shell lines are commented out. Run with a normal Python interpreter."""


# --- code cell ---

# Install required packages
# !pip install transformers datasets torch  # Jupyter-only


# --- code cell ---

import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# Disable wandb logging
os.environ["WANDB_DISABLED"] = "true"


# --- code cell ---

# Generate synthetic time series data
np.random.seed(42)
n_samples = 100
n_timestamps = 50

# Create time series with two classes
class_0 = np.random.normal(0, 1, (n_samples // 2, n_timestamps))
class_1 = np.random.normal(2, 1, (n_samples // 2, n_timestamps))

# Combine data and labels
X = np.vstack((class_0, class_1))
y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))

# Convert to DataFrame
df = pd.DataFrame(X)
df["label"] = y

print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nClass distribution:")
print(df["label"].value_counts())


# --- code cell ---

# Load BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def tokenize_time_series(series):
    """Convert numerical time series to tokenized text"""
    series_str = " ".join(map(str, series))
    return tokenizer(
        series_str,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
    )


# Tokenize the dataset
tokens = [tokenize_time_series(row[:-1].values) for _, row in df.iterrows()]
labels = df["label"].values

print(f"Tokenized {len(tokens)} time series")


# --- code cell ---

# Split data into train and test sets
train_tokens, test_tokens, train_labels, test_labels = train_test_split(
    tokens, labels, test_size=0.2, random_state=42
)


# Create dataset objects
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


train_dataset = TimeSeriesDataset(train_tokens, train_labels)
test_dataset = TimeSeriesDataset(test_tokens, test_labels)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")


# --- code cell ---

# Load pre-trained BERT model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    save_steps=10,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10,
    report_to="none",  # Disables wandb and other loggers
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)


# --- code cell ---

# Fine-tune the model
print("Training BERT model...")
trainer.train()

# Make predictions
predictions = trainer.predict(test_dataset)
predicted_labels = np.argmax(predictions.predictions, axis=1)

# Calculate accuracy
accuracy = accuracy_score(test_labels, predicted_labels)
print(f"\n✓ Test Accuracy: {accuracy:.2%}")
