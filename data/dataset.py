"""
MILESTONE 2 & 3: Dataset Collection, EDA, and Preprocessing
AI-Based Exam Anxiety Detector
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import Counter
import re
import os
import json

# ─────────────────────────────────────────────
# DATASET CREATION
# ─────────────────────────────────────────────

LOW_ANXIETY_SAMPLES = [
    "I feel pretty calm about the upcoming exam. I've studied well.",
    "I'm a bit nervous but mostly confident in my preparation.",
    "The exam is tomorrow and I feel ready. Reviewed all my notes.",
    "I studied consistently and feel okay about the test.",
    "I have a mild nervousness but it's manageable.",
    "Looking forward to showing what I've learned.",
    "I feel prepared and slightly excited about the exam.",
    "A little bit of tension, nothing I can't handle.",
    "I've done my best to prepare. Feeling neutral about it.",
    "Mild jitters, but I know the material well.",
    "I reviewed everything and feel good about tomorrow.",
    "Not too stressed — just a light feeling of anticipation.",
    "I'm calm, got enough sleep, and feel ready.",
    "A small flutter in my stomach but overall feeling fine.",
    "I have everything ready for the exam. Minimal stress.",
    "I trust my preparation. Feeling relatively at ease.",
    "The exam doesn't worry me much. I've prepared thoroughly.",
    "I'm a bit unsure on two topics but otherwise well prepared.",
    "Slight nervousness is normal, I think. I'll do fine.",
    "Reviewed flashcards and feel confident about the content.",
]

MODERATE_ANXIETY_SAMPLES = [
    "I'm somewhat anxious about the exam. Worried I may have missed some topics.",
    "The exam tomorrow is making me uneasy. I feel unsettled.",
    "I've been stressed about this test. There's a lot riding on it.",
    "My heart races a bit when I think about the exam tomorrow.",
    "I'm worried I haven't covered everything that might come up.",
    "Feeling tense and distracted. Can't focus well on my revision.",
    "There are parts of the syllabus I'm not confident about.",
    "The closer it gets, the more nervous I feel about the exam.",
    "I keep second-guessing my answers in practice tests. Stressful.",
    "I've been struggling to sleep thinking about the upcoming test.",
    "Mild panic when I can't recall formulas during revision.",
    "I feel pressure from my family to perform well in the exam.",
    "I keep going over the same notes but still feel uncertain.",
    "Anxious thoughts keep interrupting my study sessions.",
    "I'm worried about running out of time during the exam.",
    "Moderate stress — I know some of it but fear surprises.",
    "I feel overwhelmed by the volume of material to cover.",
    "There's a knot in my stomach every time I think about it.",
    "My concentration has been poor because of exam stress.",
    "I've been irritable and distracted since exam week began.",
]

HIGH_ANXIETY_SAMPLES = [
    "I'm absolutely terrified about the exam. I can't stop shaking.",
    "I feel like I'm going to fail no matter how much I study.",
    "I can't eat or sleep properly because of exam anxiety.",
    "My mind goes completely blank when I try to study. I'm panicking.",
    "I'm convinced I'm going to fail and disappoint everyone.",
    "I feel sick to my stomach every time I think about the exam.",
    "I've been crying every night because of exam stress.",
    "I'm so scared I can't think straight. Nothing is sticking.",
    "I'm in complete panic mode. Can't stop catastrophizing.",
    "The exam has been giving me chest tightness and headaches.",
    "I feel hopeless. I don't think I can pass no matter what.",
    "My hands tremble whenever I sit down to study for the exam.",
    "I keep having nightmares about failing and freezing in the hall.",
    "Every time I open my books I feel a wave of nausea.",
    "I'm spiraling — convinced I know nothing and will fail.",
    "I've stopped eating properly because anxiety is so overwhelming.",
    "Constant dread. I feel paralyzed and can't move forward.",
    "I feel like I'm going to have a breakdown before the exam.",
    "My brain won't retain anything. The pressure is unbearable.",
    "I can't stop crying. The exam feels impossible and I'm terrified.",
]

def create_dataset():
    texts, labels = [], []
    for t in LOW_ANXIETY_SAMPLES:
        texts.append(t); labels.append("Low Anxiety")
    for t in MODERATE_ANXIETY_SAMPLES:
        texts.append(t); labels.append("Moderate Anxiety")
    for t in HIGH_ANXIETY_SAMPLES:
        texts.append(t); labels.append("High Anxiety")

    df = pd.DataFrame({"text": texts, "label": labels})
    label_map = {"Low Anxiety": 0, "Moderate Anxiety": 1, "High Anxiety": 2}
    df["label_id"] = df["label"].map(label_map)
    return df


# ─────────────────────────────────────────────
# TEXT PREPROCESSING
# ─────────────────────────────────────────────

def preprocess_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z0-9\s\.\,\!\?\'\"]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def run_eda(df: pd.DataFrame, save_dir: str = "."):
    os.makedirs(save_dir, exist_ok=True)

    # ── Plot 1: Label Distribution ──────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#161b22")

    colors = ["#2dd4bf", "#f59e0b", "#f87171"]
    counts = df["label"].value_counts()

    bars = axes[0].bar(counts.index, counts.values, color=colors, edgecolor="none", width=0.55)
    axes[0].set_title("Anxiety Level Distribution", color="white", fontsize=14, pad=12)
    axes[0].set_xlabel("Anxiety Level", color="#9ca3af")
    axes[0].set_ylabel("Count", color="#9ca3af")
    axes[0].tick_params(colors="white")
    for spine in axes[0].spines.values():
        spine.set_visible(False)
    for bar, val in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, str(val),
                     ha="center", va="bottom", color="white", fontweight="bold")

    wedges, texts, autotexts = axes[1].pie(
        counts.values, labels=counts.index, colors=colors,
        autopct="%1.0f%%", startangle=140,
        textprops={"color": "white", "fontsize": 11},
        wedgeprops={"edgecolor": "#0d1117", "linewidth": 2}
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontweight("bold")
    axes[1].set_title("Label Proportion", color="white", fontsize=14, pad=12)

    plt.suptitle("EDA — Exam Anxiety Dataset", color="white", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(save_dir, "eda_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"[EDA] Saved: {path}")

    # ── Plot 2: Text Length Distribution ────────
    df["text_length"] = df["text"].apply(len)
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    for label, color in zip(["Low Anxiety", "Moderate Anxiety", "High Anxiety"], colors):
        subset = df[df["label"] == label]["text_length"]
        ax.hist(subset, bins=12, alpha=0.75, label=label, color=color, edgecolor="none")

    ax.set_title("Text Length Distribution by Anxiety Level", color="white", fontsize=14)
    ax.set_xlabel("Character Count", color="#9ca3af")
    ax.set_ylabel("Frequency", color="#9ca3af")
    ax.tick_params(colors="white")
    ax.legend(frameon=False, labelcolor="white")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.axvline(df["text_length"].mean(), color="white", linestyle="--", alpha=0.5, label="Mean")

    plt.tight_layout()
    path2 = os.path.join(save_dir, "eda_text_length.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"[EDA] Saved: {path2}")

    # ── Summary Stats ────────────────────────────
    print("\n" + "="*50)
    print("  DATASET SUMMARY")
    print("="*50)
    print(f"  Total samples     : {len(df)}")
    print(f"  Unique labels     : {df['label'].nunique()}")
    print(f"  Avg text length   : {df['text_length'].mean():.1f} chars")
    print(f"  Min text length   : {df['text_length'].min()}")
    print(f"  Max text length   : {df['text_length'].max()}")
    print(f"\n  Class distribution:")
    for label, count in counts.items():
        print(f"    {label:<20} : {count} samples")
    print("="*50 + "\n")

    return df


if __name__ == "__main__":
    df = create_dataset()
    df["text"] = df["text"].apply(preprocess_text)
    df = run_eda(df, save_dir="data")
    df.to_csv("data/exam_anxiety_dataset.csv", index=False)
    print("[✓] Dataset saved to data/exam_anxiety_dataset.csv")
