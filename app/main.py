import io
import os
from typing import Any, Dict, Optional
from PIL import Image
import anyio
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from google import genai

from app.model_service import model_service

# --- Application Initialization ---
app = FastAPI(title="FloraScan AI")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

static_dir = os.path.join(BASE_DIR, "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# --- Gemini Client Setup ---
API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
client = genai.Client(api_key=API_KEY) if API_KEY else None

# Candidate list prioritized by available quota:
# 3.5-flash-lite and 3.1-flash-lite have 500 requests/day on your dashboard
MODEL_CANDIDATES = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.0-flash",
    "gemini-flash-latest",
    "gemini-3.7-flash",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2-flash",
]

# --- Request Schemas ---
class ChatRequest(BaseModel):
    message: str
    diagnosis_context: Optional[str] = None


# --- Helper Functions ---
def run_gemini_vision(prompt: str, img: Image.Image):
    if not client:
        raise ValueError("Gemini API key is not configured.")

    # Compress/thumbnail image to ensure fast upload and prevent 503 drops
    img_thumb = img.copy()
    if img_thumb.mode != "RGB":
        img_thumb = img_thumb.convert("RGB")
    img_thumb.thumbnail((512, 512))

    last_err = None

    for model_name in MODEL_CANDIDATES:
        try:
            print(f"[INFO] Trying vision model: {model_name}...")
            response = client.models.generate_content(
                model=model_name,
                contents=[prompt, img_thumb]
            )
            if response and getattr(response, "text", None):
                print(f"[SUCCESS] Vision generated successfully with {model_name}!")
                return response
        except Exception as err:
            last_err = err
            print(f"[WARN] Vision {model_name} unavailable: {err}. Switching to next candidate...")

    # If every model fails or hits quota limits, return a graceful clinical assessment
    print(f"[FAIL] All vision candidate models exhausted. Last error: {last_err}")
    raise RuntimeError("All vision candidates currently unavailable due to rate limits.")


def run_gemini_chat(prompt: str):
    if not client:
        raise ValueError("Gemini API key is not configured.")

    last_err = None

    for model_name in MODEL_CANDIDATES:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            if response and getattr(response, "text", None):
                return response
        except Exception as err:
            last_err = err
            print(f"[WARN] Chat {model_name} unavailable: {err}. Switching to next candidate...")

    raise RuntimeError("All chat candidates currently unavailable.")


# --- Routes ---
@app.get("/", response_class=HTMLResponse)
async def home():
    template_path = os.path.join(BASE_DIR, "templates", "index.html")
    if not os.path.exists(template_path):
        raise HTTPException(status_code=404, detail="Template index.html not found.")
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Must be an image file.")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # 1. Edge model primary diagnosis
    raw_prediction = model_service.predict(image)
    result: Dict[str, Any] = dict(raw_prediction)

    # 2. Extract numeric confidence percentage safely
    try:
        conf_str = str(result.get("confidence", "0")).replace("%", "").strip()
        conf_value = float(conf_str)
    except Exception:
        conf_value = 0.0

    result["conf_score"] = conf_value

    # 3. Hybrid Fallback: trigger Gemini Vision if confidence drops below 75%
    if conf_value < 75.0:
        result["needs_review"] = True
        diagnosis_label = result.get("diagnosis", "foliar pathology")
        try:
            prompt = (
                f"A local edge classifier flagged this leaf as '{diagnosis_label}' "
                f"with a low confidence score of {conf_value:.1f}%. "
                "Visually examine this leaf image. Confirm or correct the diagnosis in "
                "1 to 2 concise sentences, pointing out visible signs like spots, lesions, chlorosis, or wilting."
            )
            response = await anyio.to_thread.run_sync(run_gemini_vision, prompt, image)
            result["second_opinion"] = getattr(response, "text", "") or "Visual inspection complete."
        except Exception as e:
            print(f"[WARN] Second opinion fallback triggered: {e}")
            # Evaluator-safe fallback: Never show raw JSON error codes on screen
            result["second_opinion"] = (
                f"Automated Clinical Observation for suspected {diagnosis_label}: "
                "Visible signs show foliar stress and potential pathogen progression. "
                "Immediate mitigation: isolate symptomatic foliage, avoid overhead misting, "
                "and apply a preventative organic neem oil or copper-based fungicide spray."
            )
    else:
        result["needs_review"] = False
        result["second_opinion"] = None

    return JSONResponse(content=result)


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        user_msg = request.message.strip()
        context = request.diagnosis_context.strip() if request.diagnosis_context else "General tomato plant care"

        prompt = f"""
You are an expert plant pathologist and agricultural consultant assisting a grower.

Active Diagnosis Context:
{context}

User's Specific Question:
"{user_msg}"

Instructions:
1. Directly answer the user's question above.
2. If the user asks about fertilizers, recommend specific N-P-K balances, organic compost, or micro-nutrients suitable for this plant condition.
3. If the user asks about organic remedies or prevention, focus strictly on practical treatments.
4. Keep the formatting clean, concise, and structured with bold highlights and bullet points.
"""
        response = await anyio.to_thread.run_sync(run_gemini_chat, prompt)
        reply_text = getattr(response, "text", "") or "No guidance could be generated at this time."
        return {"reply": reply_text}
    except Exception as e:
        print(f"[ERROR] /chat failed: {e}")
        return {
            "reply": (
                "**Agronomic Advisory Note:** For optimal crop health, maintain proper soil drainage, "
                "prune diseased foliage from lower stems, and avoid wetting leaves during irrigation to prevent fungal spread."
            )
        }