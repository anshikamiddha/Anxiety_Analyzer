"""
anxiety_analyzer.py
Backend module for AI-Based Exam Anxiety Detection System
Uses Groq API with LLaMA model and a clinical system prompt.
API key is loaded automatically from a .env file in the project root.
"""

import os
import json
import re
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY not found. Make sure your .env file exists and contains:\n"
        "GROQ_API_KEY=gsk_your_key_here"
    )

SYSTEM_PROMPT = """You are an empathetic AI psychologist specializing in academic and exam-related anxiety detection. 
Your role is to analyze student messages and assess their anxiety levels based on their language patterns, emotional cues, 
cognitive distortions, physiological symptoms mentioned, and behavioral indicators.

For every student message you receive, you MUST respond in the following strict JSON format:
{
  "anxiety_score": <integer from 0 to 100>,
  "anxiety_level": "<one of: Minimal, Mild, Moderate, High, Severe>",
  "emotional_tone": "<brief descriptor e.g. 'Overwhelmed', 'Nervous', 'Calm', 'Panicked'>",
  "key_indicators": ["<indicator 1>", "<indicator 2>", "<indicator 3>"],
  "cognitive_distortions": ["<distortion 1>", "<distortion 2>"],
  "physical_symptoms_mentioned": ["<symptom 1>", "<symptom 2>"],
  "supportive_response": "<2-3 sentence warm, empathetic, constructive response to the student>",
  "coping_strategies": ["<strategy 1>", "<strategy 2>", "<strategy 3>"],
  "risk_flags": ["<any red flags if present, else empty list>"]
}

Scoring Guide:
- 0-20: Minimal anxiety — student appears calm, confident, maybe slight nerves
- 21-40: Mild anxiety — some worry present but manageable, student is coping
- 41-60: Moderate anxiety — noticeable distress, affecting focus/sleep/confidence
- 61-80: High anxiety — significant impairment, panic-like thinking, avoidance behaviors
- 81-100: Severe anxiety — crisis-level, mentions of breakdown, inability to function

Always be compassionate. Never be dismissive. Identify patterns like catastrophizing, 
all-or-nothing thinking, mind-reading, fortune-telling, and emotional reasoning.

IMPORTANT: Respond ONLY with valid JSON. Do not include markdown code blocks or any text outside the JSON."""


def analyze_anxiety(user_message: str, api_key: str = None, model: str = "llama3-70b-8192") -> dict:
    """
    Sends user message to Groq API and returns structured anxiety analysis.

    Args:
        user_message: The student's text input
        api_key: (Optional) Groq API key override. Defaults to GROQ_API_KEY from .env
        model: Groq model to use

    Returns:
        dict with full anxiety analysis
    """
    key = api_key or GROQ_API_KEY
    # Default to a supported model if the provided model is decommissioned
    # supported_models = ["llama3-8b-8192", "mixtral-8x7b-32768", "gemma2-9b-it"]
    model = "llama-3.1-8b-instant"
    client = Groq(api_key=key)

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3,
            max_tokens=1024,
        )

        raw_response = completion.choices[0].message.content.strip()

        # Clean up if model wraps response in markdown code fences
        raw_response = re.sub(r"```json|```", "", raw_response).strip()

        result = json.loads(raw_response)
        result["raw_input"] = user_message
        result["model_used"] = model
        return result

    except json.JSONDecodeError as e:
        return {
            "error": f"Failed to parse model response: {str(e)}",
            "raw_response": raw_response if "raw_response" in locals() else "No response",
            "anxiety_score": 0,
            "anxiety_level": "Unknown",
            "emotional_tone": "Unknown",
            "key_indicators": [],
            "cognitive_distortions": [],
            "physical_symptoms_mentioned": [],
            "supportive_response": "We encountered an issue analyzing your message. Please try again.",
            "coping_strategies": [],
            "risk_flags": [],
            "raw_input": user_message,
        }
    except Exception as e:
        return {
            "error": str(e),
            "anxiety_score": 0,
            "anxiety_level": "Error",
            "emotional_tone": "Unknown",
            "key_indicators": [],
            "cognitive_distortions": [],
            "physical_symptoms_mentioned": [],
            "supportive_response": f"An error occurred: {str(e)}",
            "coping_strategies": [],
            "risk_flags": [],
            "raw_input": user_message,
        }
