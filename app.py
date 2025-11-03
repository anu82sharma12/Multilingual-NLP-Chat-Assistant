
---

## app.py

from flask import Flask, request, jsonify, render_template
import torch
from transformers import MT5ForConditionalGeneration, T5Tokenizer
import joblib
import json
import re

app = Flask(__name__)

# Load models
tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
translator = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
intent_model = joblib.load("models/intent_classifier.pkl")
ner_model = joblib.load("models/ner_model/ner.pkl")

# Responses
with open("responses.json") as f:
    RESPONSES = json.load(f)

# Language map
LANG_MAP = {
    "hi": "Hindi", "ta": "Tamil", "te": "Telugu",
    "bn": "Bengali", "mr": "Marathi", "en": "English"
}

def detect_language(text):
    hindi = bool(re.search("[\u0900-\u097F]", text))
    tamil = bool(re.search("[\u0B80-\u0BFF]", text))
    telugu = bool(re.search("[\u0C00-\u0C7F]", text))
    bengali = bool(re.search("[\u0980-\u09FF]", text))
    marathi = hindi or bool(re.search("[\u0900-\u097F]", text))
    return "hi" if hindi else "ta" if tamil else "te" if telugu else "bn" if bengali else "mr" if marathi else "en"

def translate(text, src, tgt):
    input_text = f"translate {LANG_MAP[src]} to {LANG_MAP[tgt]}: {text}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True)
    outputs = translator.generate(**inputs, max_length=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["message"]
    src_lang = detect_language(user_input)

    # Translate to English
    if src_lang != "en":
        eng_input = translate(user_input, src_lang, "en")
    else:
        eng_input = user_input

    # Intent + NER
    intent = intent_model.predict([eng_input])[0]
    entities = ner_model.predict([eng_input])[0]

    # Generate English response
    response_key = f"{intent}_{entities[0] if entities else 'none'}"
    eng_response = RESPONSES.get(response_key, RESPONSES["fallback"])

    # Back-translate
    if src_lang != "en":
        final_response = translate(eng_response, "en", src_lang)
    else:
        final_response = eng_response

    return jsonify({
        "response": final_response,
        "intent": intent,
        "language": LANG_MAP[src_lang]
    })

if __name__ == "__main__":
    app.run(debug=True)
