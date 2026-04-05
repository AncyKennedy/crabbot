from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import os
from tensorflow import keras
from google import genai as google_genai

# ─── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH  = "C:/Users/HP/Music/crab/crab_model.keras"                              # your model file
IMG_SIZE    = (160, 160)                                      # must match training
CLASS_NAMES = ["flower_crab", "mud_crab", "three_dot_crab"]  # same order as training

# ─── Gemini setup ──────────────────────────────────────────────────────────────
GEMINI_KEY    = os.environ.get("GEMINI_API_KEY", "AIzaSyAbokOCuAOjh8BKJocta3F2slrA9U8z7ZQ")
google_client = google_genai.Client(api_key=GEMINI_KEY)

SYSTEM_PROMPT = """You are CrabBot, a friendly crab expert specializing in Tamil Nadu, India crab species.
You deeply know these three species:
1. Flower Crab (Portunus pelagicus / பூ நண்டு / Poo Nandu)
2. Mud Crab (Scylla serrata / சேற்று நண்டு / Shetru Nandu)
3. Three-Spot Crab (Portunus sanguinolentus / மூன்று புள்ளி நண்டு / Moonru Pulli Nandu)

Answer questions about crab biology, habitat, diet, behavior, fishing, cooking, nutrition, and Tamil Nadu crab culture.
Be friendly and conversational. Keep answers to 3-5 sentences.
If the user writes in Tamil, reply in Tamil.
If asked about something unrelated to crabs, politely redirect to crab topics."""

# ─── Load model ────────────────────────────────────────────────────────────────
model = None

def load_model():
    global model
    model = keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded:", MODEL_PATH)
    print("   Classes:", CLASS_NAMES)

def preprocess_image(image_bytes):
    """Preprocess image to match training pipeline exactly."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img) / 255.0        # normalize to [0, 1]
    arr = np.expand_dims(arr, axis=0)  # → (1, 160, 160, 3)
    return arr

# ─── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    image_bytes = request.files["image"].read()

    try:
        img_array = preprocess_image(image_bytes)
    except Exception as e:
        return jsonify({"error": f"Image processing failed: {str(e)}"}), 400

    try:
        predictions     = model.predict(img_array)
        predicted_index = int(np.argmax(predictions[0]))
        confidence      = float(np.max(predictions[0])) * 100
        label           = CLASS_NAMES[predicted_index]
        return jsonify({
            "label":      label,
            "confidence": round(confidence, 2),
            "all_scores": {
                CLASS_NAMES[i]: round(float(v) * 100, 2)
                for i, v in enumerate(predictions[0])
            }
        })
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route("/chat", methods=["POST"])
def chat():
    question = request.json.get("question", "").strip()
    if not question:
        return jsonify({"error": "No question provided."}), 400
    try:
        response = google_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"{SYSTEM_PROMPT}\n\nUser question: {question}"
        )
        return jsonify({"answer": response.text})
    except Exception as e:
        return jsonify({"error": f"Gemini error: {str(e)}"}), 500

# ─── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=7860)  # 7860 for Hugging Face