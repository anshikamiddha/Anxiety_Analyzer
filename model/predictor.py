"""
MILESTONE 6: Anxiety Prediction Logic
AI-Based Exam Anxiety Detector
"""

import os
import re
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from typing import Dict, List, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
LABEL_MAP = {0: "Low Anxiety", 1: "Moderate Anxiety", 2: "High Anxiety"}

ANXIETY_EMOJIS = {
    "Low Anxiety": "😊",
    "Moderate Anxiety": "😟",
    "High Anxiety": "😰",
}

ANXIETY_COLORS = {
    "Low Anxiety": "#2dd4bf",
    "Moderate Anxiety": "#f59e0b",
    "High Anxiety": "#f87171",
}

ANXIETY_TIPS = {
    "Low Anxiety": [
        "✅ Your stress levels appear manageable. Keep up your preparation routine!",
        "🧘 Take short breaks every 45 minutes to stay refreshed.",
        "💤 Ensure you get 7–8 hours of sleep the night before.",
        "🍎 Eat balanced meals to keep your energy stable.",
        "🎯 Review key topics briefly the morning of the exam.",
    ],
    "Moderate Anxiety": [
        "🌬️ Try deep breathing: inhale for 4s, hold 4s, exhale 4s. Repeat 5×.",
        "📝 Break your revision into small, manageable 20-minute chunks.",
        "🏃 Light exercise (a 15-min walk) can reduce stress hormones.",
        "🗣️ Talk to a friend, classmate, or teacher about your concerns.",
        "📋 Write down your worries — externalizing them reduces their grip.",
        "🎵 Calm instrumental music while studying can help focus.",
        "⏰ Avoid cramming; trust the preparation you've already done.",
    ],
    "High Anxiety": [
        "🆘 You may be experiencing significant exam anxiety. Please consider speaking with a school counselor or trusted adult.",
        "💆 Try the 5-4-3-2-1 grounding technique: name 5 things you see, 4 you hear, 3 you feel, 2 you smell, 1 you taste.",
        "🌬️ Practice box breathing: inhale 4s → hold 4s → exhale 4s → hold 4s.",
        "📵 Step away from studying for 30 minutes — rest is NOT wasted time.",
        "💬 Share how you feel with someone you trust immediately.",
        "🧊 Splash cold water on your face to activate your calm reflex.",
        "🌿 Remember: one exam does not define your worth or your future.",
        "📞 Contact your institution's student support services if available.",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# RULE-BASED FALLBACK (when model not available)
# ─────────────────────────────────────────────────────────────────────────────
HIGH_ANXIETY_KEYWORDS = [
    "terrified", "panic", "crying", "can't stop", "breakdown", "hopeless",
    "shaking", "trembling", "nausea", "sick", "nightmare", "paralyzed",
    "unbearable", "chest", "spiral", "catastroph", "nothing sticking",
    "complete panic", "absolute terror", "feel like failing",
]

MODERATE_ANXIETY_KEYWORDS = [
    "worried", "anxious", "nervous", "uneasy", "stressed", "tense",
    "distracted", "overwhelmed", "unsure", "second-guessing", "racing",
    "can't focus", "pressure", "fear", "kn0t", "struggling",
]

LOW_ANXIETY_KEYWORDS = [
    "calm", "confident", "ready", "prepared", "okay", "fine", "manageable",
    "okay", "neutral", "at ease", "slightly", "mild", "small",
]


def rule_based_predict(text: str) -> Tuple[str, float, Dict]:
    text_lower = text.lower()
    high_score = sum(1 for kw in HIGH_ANXIETY_KEYWORDS if kw in text_lower)
    mod_score = sum(1 for kw in MODERATE_ANXIETY_KEYWORDS if kw in text_lower)
    low_score = sum(1 for kw in LOW_ANXIETY_KEYWORDS if kw in text_lower)

    scores = np.array([low_score + 1, mod_score + 1, high_score + 1], dtype=float)
    probs = scores / scores.sum()

    predicted_idx = int(np.argmax(probs))
    label = LABEL_MAP[predicted_idx]
    confidence = float(probs[predicted_idx])

    return label, confidence, {
        "Low Anxiety": round(float(probs[0]), 4),
        "Moderate Anxiety": round(float(probs[1]), 4),
        "High Anxiety": round(float(probs[2]), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# BERT PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────
class AnxietyPredictor:
    def __init__(self, model_path: str = "model/anxiety_bert_model", use_gpu: bool = True):
        self.model_path = model_path
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self._loaded = False
        self._try_load()

    def _try_load(self):
        if os.path.exists(self.model_path):
            try:
                print(f"[Model] Loading BERT from {self.model_path}...")
                self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
                self.model = BertForSequenceClassification.from_pretrained(self.model_path)
                self.model.to(self.device)
                self.model.eval()
                self._loaded = True
                print(f"[Model] Loaded on {self.device}")
            except Exception as e:
                print(f"[Model] Load failed: {e}. Falling back to rule-based.")
        else:
            print("[Model] No trained model found. Using rule-based predictor.")

    def preprocess(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9\s\.\,\!\?\'\"]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def predict(self, text: str) -> Dict:
        text = self.preprocess(text)

        if not self._loaded:
            label, confidence, all_probs = rule_based_predict(text)
        else:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=128,
                padding="max_length",
                truncation=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1).squeeze().cpu().numpy()

            predicted_idx = int(np.argmax(probs))
            label = LABEL_MAP[predicted_idx]
            confidence = float(probs[predicted_idx])
            all_probs = {LABEL_MAP[i]: round(float(p), 4) for i, p in enumerate(probs)}

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "probabilities": all_probs,
            "emoji": ANXIETY_EMOJIS[label],
            "color": ANXIETY_COLORS[label],
            "tips": ANXIETY_TIPS[label],
            "model_type": "BERT" if self._loaded else "Rule-Based",
            "disclaimer": (
                "⚠️ This tool is non-diagnostic and for supportive purposes only. "
                "It does not replace professional mental health assessment."
            ),
        }

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        return [self.predict(t) for t in texts]


# ─────────────────────────────────────────────────────────────────────────────
# Singleton instance
# ─────────────────────────────────────────────────────────────────────────────
_predictor: AnxietyPredictor = None


def get_predictor() -> AnxietyPredictor:
    global _predictor
    if _predictor is None:
        _predictor = AnxietyPredictor()
    return _predictor


if __name__ == "__main__":
    p = get_predictor()
    test_texts = [
        "I feel calm and ready for the exam tomorrow.",
        "I'm worried I haven't covered everything and feel tense.",
        "I'm absolutely terrified. I can't stop shaking and I feel sick.",
    ]
    for t in test_texts:
        result = p.predict(t)
        print(f"\nInput   : {t[:60]}...")
        print(f"Label   : {result['emoji']} {result['label']}")
        print(f"Confidence : {result['confidence']:.2%}")
        print(f"Probs   : {result['probabilities']}")
