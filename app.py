"""
app.py — Kisha-Tech Electronics AI Backend
Hosted on Render at: kisha-tech-backend.onrender.com

Endpoints:
    POST /ask         — AI query from frontend
    GET  /health      — strict uptime check
    GET  /health/full — detailed diagnostics (debug only)
"""

import os
import time
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("kishatech.backend")

# ── Env validation ────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    logger.error("FATAL: GROQ_API_KEY not set.")
    raise SystemExit("GROQ_API_KEY required")

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

logger.info("=== STARTUP CHECK ===")
logger.info(f"Groq: {'✓ configured' if GROQ_API_KEY else '✗ missing'}")
logger.info(f"Supabase: {'✓ configured' if SUPABASE_URL else '✗ not set'}")

# ── Flask + CORS ──────────────────────────────────────────────────────────────
app = Flask(__name__)

CORS(
    app,
    origins=[
        "https://kishatechadmin.vercel.app",
        "https://kishatech.vercel.app"
    ],
    supports_credentials=True
)

# ── Request Logging Middleware ────────────────────────────────────────────────
@app.before_request
def log_request():
    logger.info(
        f"{request.method} {request.path} | "
        f"IP={request.remote_addr} | "
        f"UA={request.headers.get('User-Agent', '')[:60]}"
    )

# ── Groq Client ───────────────────────────────────────────────────────────────
client = Groq(api_key=GROQ_API_KEY)
MODEL  = "llama-3.3-70b-versatile"

SYSTEM_BASE = """You are Sarah, the AI assistant for Kisha-Tech Electronics & Hardware Store.
Location: Machakos Kenya Israel, opposite Manza College.
Hours: Mon–Sat 7:00 AM – 7:00 PM, Sun 9:00 AM – 5:00 PM.

You are the shop's financial and inventory advisor.
Be direct, specific, and always use KSh figures.
Answer in plain English. Under 300 words unless needed.
Never make up inventory items or prices — only use provided context."""

# ── /ask ──────────────────────────────────────────────────────────────────────
@app.route("/ask", methods=["POST"])
def ask():
    start_time = time.time()

    try:
        if not request.is_json:
            logger.warning("/ask invalid content-type")
            return jsonify({"error": "Content-Type must be application/json"}), 400

        data = request.get_json(silent=True) or {}

        message = (data.get("message") or "").strip()
        context = (data.get("context") or "").strip()
        history = data.get("history") or []

        if not message:
            logger.warning("/ask missing message")
            return jsonify({"error": "message is required"}), 400

        logger.info(f"/ask | msg='{message[:50]}' | ctx={len(context)}")

        # Build system prompt
        system_content = SYSTEM_BASE
        if context:
            system_content += f"\n\n=== SHOP DATA ===\n{context[:4000]}"

        messages = [{"role": "system", "content": system_content}]

        for turn in history[-10:]:
            role = turn.get("role")
            content = str(turn.get("content", ""))[:600]
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": message})

        completion = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=600,
            timeout=15,
        )

        reply = completion.choices[0].message.content.strip()

        duration = round((time.time() - start_time) * 1000)

        logger.info(f"/ask success | {duration}ms | reply='{reply[:80]}'")

        return jsonify({
            "reply": reply,
            "latency_ms": duration
        })

    except Exception as e:
        duration = round((time.time() - start_time) * 1000)
        logger.exception(f"/ask failure | {duration}ms")

        return jsonify({
            "error": "AI service temporarily unavailable",
            "latency_ms": duration
        }), 500

# ── /health (STRICT — frontend + Render) ──────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok"
    }), 200

# ── /health/full (DEBUG ONLY) ─────────────────────────────────────────────────
@app.route("/health/full", methods=["GET"])
def health_full():
    return jsonify({
        "status": "ok",
        "service": "kisha-tech-backend",
        "model": MODEL,
        "groq": "configured" if GROQ_API_KEY else "missing",
        "supabase": "configured" if SUPABASE_URL else "not set",
    }), 200

# ── Root ──────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "message": "Kisha-Tech AI Backend — Sarah is live"
    })

# ── Startup ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    logger.info(f"Starting server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)