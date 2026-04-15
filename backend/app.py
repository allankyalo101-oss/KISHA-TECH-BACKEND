# ── LOAD ENV (BULLETPROOF) ───────────────────────────────────
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# ── IMPORTS ──────────────────────────────────────────────────
from flask import Flask, request, jsonify
import requests
from flask_cors import CORS
from supabase import create_client

# ── INIT APP ─────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── ENV CONFIG ───────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Debug (remove later)
print("GROQ:", str(GROQ_API_KEY)[:10])
print("SUPABASE:", str(SUPABASE_KEY)[:10])

# Validate environment
if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not set in .env")

if not SUPABASE_URL:
    raise ValueError("❌ SUPABASE_URL not set in .env")

if not SUPABASE_KEY:
    raise ValueError("❌ SUPABASE_KEY not set in .env")

# ── INIT SUPABASE ────────────────────────────────────────────
try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    raise RuntimeError(f"❌ Supabase init failed: {str(e)}")

# ── LOAD INVENTORY ───────────────────────────────────────────
def get_inventory():
    try:
        res = supabase.table("inventory").select("*").execute()
        return res.data if res.data else []
    except Exception as e:
        print("Inventory fetch error:", str(e))
        return []

# ── COMPUTE METRICS (CRITICAL UPGRADE) ───────────────────────
def compute_metrics(items):
    try:
        total_items = len(items)
        total_stock_value = sum((i.get("sell_price", 0) or 0) * (i.get("qty", 0) or 0) for i in items)
        low_stock = [i for i in items if (i.get("qty", 0) or 0) <= 5]

        return {
            "total_items": total_items,
            "total_stock_value": total_stock_value,
            "low_stock_count": len(low_stock),
        }
    except Exception as e:
        print("Metrics error:", str(e))
        return {}

# ── AI ENDPOINT ──────────────────────────────────────────────
@app.route("/ask", methods=["POST"])
def ask_ai():
    try:
        body = request.get_json()
        user_input = (body.get("message") or "").strip() if body else ""

        if not user_input:
            return jsonify({"error": "Message cannot be empty"}), 400

        inventory = get_inventory()
        metrics = compute_metrics(inventory)

        # Limit dataset (token efficiency)
        sample = inventory[:50]

        context = f"""
You are a professional electronics shop analyst.

Business metrics:
{metrics}

Inventory sample:
{sample}

Rules:
- Be concise
- Use numbers
- Give actionable advice
"""

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": context},
                    {"role": "user", "content": user_input}
                ]
            },
            timeout=30
        )

        # ── HANDLE API FAILURE ───────────────────────────────
        if response.status_code != 200:
            print("Groq error:", response.text)
            return jsonify({
                "error": f"Groq API error {response.status_code}"
            }), 500

        data = response.json()

        # Safe extraction
        reply = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "No response from AI")
        )

        return jsonify({"reply": reply})

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Request failed: {str(e)}"}), 500

    except Exception as e:
        print("Server error:", str(e))
        return jsonify({"error": f"Server error: {str(e)}"}), 500

# ── HEALTH CHECK (IMPORTANT FOR DEPLOYMENT) ──────────────────
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "AI backend running"})

# ── RUN ──────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)