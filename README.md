# AI-Based Exam Anxiety Detector (AnxietyLens)

An interactive Streamlit app that analyzes exam-related student text and returns a structured anxiety assessment using  using finetuning the bert model with go emotions dataset.

## What It Does
- Accepts a student message about exam stress
- Generates an anxiety score (`0-100`) and level (`Minimal`, `Mild`, `Moderate`, `High`, `Severe`)
- Detects emotional tone, indicators, distortions, and physical symptoms
- Provides a supportive response and coping strategies
- Tracks session history with charts (timeline, gauge, distribution)

## Current Runtime Flow (from code)
- UI entrypoint: `app.py` (root)
- Analysis module: `anxiety_analyzer.py`


## Output Schema
`analyze_anxiety()` returns a dictionary with:
- `anxiety_score`
- `anxiety_level`
- `emotional_tone`
- `key_indicators`
- `cognitive_distortions`
- `physical_symptoms_mentioned`
- `supportive_response`
- `coping_strategies`
- `risk_flags`
- `raw_input`
- `model_used`


## Run
```bash
streamlit run app.py
```

## Project Structure (relevant files)
```text
AI-BASED-EXAM-ANXIETY-DETECTOR/
├── app.py                    # Main Streamlit UI used for runtime
├── anxiety_analyzer.py       # Groq-based anxiety analysis logic
├── .env                      # GROQ_API_KEY
├── requirements.txt
├── data/
├── model/
├── backend/
└── frontend/
```

## Notes
- Some labels in the UI still mention BERT/FastAPI, but runtime analysis is currently through Groq + LLaMA from `anxiety_analyzer.py`.
- This is a supportive educational tool, not a clinical diagnosis system.
