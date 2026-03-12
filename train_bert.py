"""
MILESTONE 4 & 5: BERT Model Training and Evaluation
AI-Based Exam Anxiety Detector

Run this on Google Colab with GPU runtime:
  Runtime > Change runtime type > T4 GPU
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    "model_name": "bert-base-uncased",
    "num_labels": 3,
    "max_length": 128,
    "batch_size": 16,
    "num_epochs": 5,
    "learning_rate": 2e-5,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "test_size": 0.2,
    "random_seed": 42,
    "model_save_path": "model/anxiety_bert_model",
    "label_names": ["Low Anxiety", "Moderate Anxiety", "High Anxiety"],
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] Using: {DEVICE}")
torch.manual_seed(CONFIG["random_seed"])
np.random.seed(CONFIG["random_seed"])


# ─────────────────────────────────────────────────────────────────────────────
# DATASET CLASS
# ─────────────────────────────────────────────────────────────────────────────
class AnxietyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch in tqdm(loader, desc="  Training", leave=False):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


def evaluate(model, loader):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="  Evaluating", leave=False):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return total_loss / len(loader), acc, f1, all_preds, all_labels


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────
def plot_training_curves(history, save_dir="."):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#161b22")

    metrics = [
        ("loss", "Train Loss", "Val Loss", "#f59e0b", "#f87171"),
        ("acc", "Train Acc", "Val Acc", "#2dd4bf", "#818cf8"),
        ("f1", None, "Val F1", None, "#34d399"),
    ]

    for i, (key, train_lbl, val_lbl, tc, vc) in enumerate(metrics):
        ax = axes[i]
        if train_lbl and f"train_{key}" in history:
            ax.plot(epochs, history[f"train_{key}"], color=tc, lw=2.5, label=train_lbl, marker="o", ms=5)
        ax.plot(epochs, history[f"val_{key}"], color=vc, lw=2.5, label=val_lbl, marker="s", ms=5)
        ax.set_title(val_lbl.replace("Val ", "") + " Curve", color="white", fontsize=13)
        ax.set_xlabel("Epoch", color="#9ca3af")
        ax.tick_params(colors="white")
        ax.legend(frameon=False, labelcolor="white")
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.suptitle("BERT Training Curves — Exam Anxiety Detector", color="white", fontsize=15, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"[✓] Saved: {path}")


def plot_confusion_matrix(y_true, y_pred, label_names, save_dir="."):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="YlOrRd",
        xticklabels=label_names, yticklabels=label_names,
        linewidths=0.5, linecolor="#0d1117",
        annot_kws={"size": 14, "weight": "bold"},
        ax=ax,
    )
    ax.set_title("Confusion Matrix", color="white", fontsize=14, pad=12)
    ax.set_xlabel("Predicted Label", color="#9ca3af", fontsize=11)
    ax.set_ylabel("True Label", color="#9ca3af", fontsize=11)
    ax.tick_params(colors="white", labelsize=10)

    path = os.path.join(save_dir, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"[✓] Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Load dataset
    df = pd.read_csv("data/exam_anxiety_dataset.csv")
    texts = df["text"].tolist()
    labels = df["label_id"].tolist()

    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=CONFIG["test_size"],
        random_state=CONFIG["random_seed"], stratify=labels
    )
    print(f"[Data] Train: {len(X_train)} | Val: {len(X_val)}")

    tokenizer = BertTokenizer.from_pretrained(CONFIG["model_name"])
    train_dataset = AnxietyDataset(X_train, y_train, tokenizer, CONFIG["max_length"])
    val_dataset = AnxietyDataset(X_val, y_val, tokenizer, CONFIG["max_length"])
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"])

    model = BertForSequenceClassification.from_pretrained(
        CONFIG["model_name"], num_labels=CONFIG["num_labels"]
    ).to(DEVICE)

    total_steps = len(train_loader) * CONFIG["num_epochs"]
    warmup_steps = int(total_steps * CONFIG["warmup_ratio"])

    optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}
    best_val_acc = 0.0

    print("\n" + "═"*50)
    print("  BERT FINE-TUNING — EXAM ANXIETY DETECTOR")
    print("═"*50)

    for epoch in range(1, CONFIG["num_epochs"] + 1):
        print(f"\n[Epoch {epoch}/{CONFIG['num_epochs']}]")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler)
        val_loss, val_acc, val_f1, preds, truths = evaluate(model, val_loader)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(f"  Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  F1: {val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(CONFIG["model_save_path"], exist_ok=True)
            model.save_pretrained(CONFIG["model_save_path"])
            tokenizer.save_pretrained(CONFIG["model_save_path"])
            print(f"  [✓] Best model saved (val_acc={val_acc:.4f})")

    # Final evaluation
    _, _, _, final_preds, final_truths = evaluate(model, val_loader)
    print("\n" + "═"*50)
    print("  CLASSIFICATION REPORT")
    print("═"*50)
    print(classification_report(final_truths, final_preds, target_names=CONFIG["label_names"]))

    # Save artifacts
    plot_training_curves(history, save_dir="model")
    plot_confusion_matrix(final_truths, final_preds, CONFIG["label_names"], save_dir="model")

    with open("model/training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n[✓] Training complete. Best Val Accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
