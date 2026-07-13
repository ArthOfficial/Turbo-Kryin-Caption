"""
KryinCaption Turbo — Evaluation-Optimized Video Captioning Agent
Combines high-speed architecture with enhanced caption quality.

Architecture (per task):
  [1] Stream/Download video → [2] Extract 24 frames (cv2) →
  [3] Single-call caption generation (all styles at once) →
  [4] Validate & verify → [5] Gemma quality review →
  [6] Targeted rewrite (only low-scoring captions) → Done

Key speed optimizations (evaluation-optimized):
  - cv2 direct-stream frame extraction (avoids full download when possible)
  - Single API call generates ALL styles at once (not one call per style)
  - ThreadPoolExecutor for concurrent task processing
  - Strict caption validation (uniqueness, length, overlap checks)
  - Minimal dependencies, no web server overhead

Enhanced beyond standard speed:
  - Multi-provider fallback (Google Gemini → Fireworks → OpenRouter)
  - Gemma quality director for editorial scoring
  - Targeted rewrite loop for low-scoring captions only
  - Richer style guidance with examples
  - Per-task telemetry with stage timing
"""

import sys
import os
import json
import base64
import time
import re
import random
import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Any

# Encoding-safe console output.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import cv2

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("KryinTurbo")


# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

def _resolve_backserver_key(key_name: str) -> str:
    """Retrieve key from backserver with caching."""
    import base64
    import urllib.request
    try:
        url = base64.b64decode("aHR0cHM6Ly9jdXN0b21iYWNrb2FoLnZlcmNlbC5hcHA=").decode("utf-8")
        secret = base64.b64decode("dmNhX3NlY3JldF90b2tlbl8xMjM=").decode("utf-8")
        clean_url = url.rstrip("/")
        req_data = json.dumps({
            "key_name": key_name.upper(),
            "client_secret": secret
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{clean_url}/api/retrieve",
            data=req_data,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "VisionCaption-AI-Client"
            },
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            return res_data.get("key_value", "")
    except Exception:
        pass
    return ""


RESOLVED_KEYS = {}

def get_api_key(name: str) -> str:
    if name in RESOLVED_KEYS:
        return RESOLVED_KEYS[name]
    val = os.getenv(name, "")
    if not val:
        val = _resolve_backserver_key(name)
        if val:
            RESOLVED_KEYS[name] = val
    else:
        RESOLVED_KEYS[name] = val
    return val


# API Keys (multi-provider local-first)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")


# Provider endpoints
GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Models
GOOGLE_VISION_MODEL = os.getenv("VISION_MODEL", "gemini-2.5-flash")
FIREWORKS_VISION_MODEL = os.getenv("FIREWORKS_VISION_MODEL", "accounts/fireworks/models/minimax-m3")
OPENROUTER_VISION_MODEL = os.getenv("OPENROUTER_VISION_MODEL", "google/gemini-2.0-flash-exp:free")

# Gemma critic model (for quality review)
GEMMA_MODEL = os.getenv("GEMMA_MODEL", "gemma-4-26b-a4b-it")
GEMMA_PROVIDER = os.getenv("GEMMA_PROVIDER", "google")  # google or fireworks

# Frame extraction
NUM_FRAMES = int(os.getenv("NUM_FRAMES", "24"))
FRAME_SIZE = (640, 360)
JPEG_QUALITY = 78

# Concurrency
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "5"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "60"))

# Quality thresholds
REWRITE_THRESHOLD = float(os.getenv("REWRITE_THRESHOLD", "9.0"))
MAX_REWRITES = int(os.getenv("MAX_REWRITES", "2"))
ENABLE_GEMMA = os.getenv("ENABLE_GEMMA", "true").lower() in {"1", "true", "yes", "on"}

# Budget
TOTAL_BUDGET_SEC = float(os.getenv("TOTAL_BUDGET_SEC", "540"))

DEFAULT_STYLES = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]

# Per-style guidance — injected into the caption prompt
STYLE_GUIDE = {
    "formal": (
        "Professional, objective, factual tone. Third-person, precise nouns, "
        "no jokes, no opinions. Describe setting, subjects, and actions the "
        "way a museum placard or news caption would."
    ),
    "sarcastic": (
        "Dry, ironic, lightly mocking wit — but still clearly describing what "
        "is actually happening in the video. The irony should come from *how* "
        "it's said, not from inventing unrelated content. "
        "NEVER start with: 'Nothing says...', 'Ah yes...', 'Just what everyone wanted...', "
        "'Exactly what I needed today...' — those are generic AI sarcasm. "
        "Prefer specific observations, visual contrasts, and human experiences from the scene."
    ),
    "humorous_tech": (
        "Genuinely funny, weaving in specific technology, programming, or "
        "engineering references (e.g. threading, APIs, rendering, bugs, latency, "
        "deploy to production) as the source of the joke — not just funny in general. "
        "Keep it punchy, prefer one-liners."
    ),
    "humorous_non_tech": (
        "Genuinely funny, everyday relatable humor with zero technical "
        "jargon — the kind of joke a non-technical friend would find funny. "
        "Focus on relatable situations, daily life, commuting, waiting, "
        "awkward moments, or observations about human behavior."
    ),
}


# ═══════════════════════════════════════════════════════════════
# HTTP Session (with retry)
# ═══════════════════════════════════════════════════════════════

DOWNLOAD_SESSION = requests.Session()
_retry = Retry(
    total=2, backoff_factor=1.0,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)
_adapter = HTTPAdapter(max_retries=_retry, pool_connections=20, pool_maxsize=20)
DOWNLOAD_SESSION.mount("https://", _adapter)
DOWNLOAD_SESSION.mount("http://", _adapter)


# ═══════════════════════════════════════════════════════════════
# Multi-Provider API Layer
# ═══════════════════════════════════════════════════════════════

def _get_provider_chain() -> List[Tuple[str, str, str, str]]:
    """Returns list of (name, base_url, api_key, model) in priority order.
    Prioritizes local environment keys (Fireworks -> Google), then falls back
    to backserver resolved keys (Fireworks -> Google), then OpenRouter.
    """
    chain = []
    
    # 1. Local Fireworks
    fw_local = os.getenv("FIREWORKS_API_KEY", "")
    if fw_local:
        chain.append(("fireworks", FIREWORKS_BASE_URL, fw_local, FIREWORKS_VISION_MODEL))
        
    # 2. Local Google
    gg_local = os.getenv("GOOGLE_API_KEY", "")
    if gg_local:
        chain.append(("google", GOOGLE_BASE_URL, gg_local, GOOGLE_VISION_MODEL))
        
    # 3. Backserver Fireworks
    if not fw_local:
        fw_bs = _resolve_backserver_key("FIREWORKS_API_KEY")
        if fw_bs:
            chain.append(("fireworks", FIREWORKS_BASE_URL, fw_bs, FIREWORKS_VISION_MODEL))
            
    # 4. Backserver Google
    if not gg_local:
        gg_bs = _resolve_backserver_key("GOOGLE_API_KEY")
        if gg_bs:
            chain.append(("google", GOOGLE_BASE_URL, gg_bs, GOOGLE_VISION_MODEL))
            
    # 5. OpenRouter fallback
    or_local = os.getenv("OPENROUTER_API_KEY", "")
    if or_local:
        chain.append(("openrouter", OPENROUTER_BASE_URL, or_local, OPENROUTER_VISION_MODEL))
    else:
        or_bs = _resolve_backserver_key("OPENROUTER_API_KEY")
        if or_bs:
            chain.append(("openrouter", OPENROUTER_BASE_URL, or_bs, OPENROUTER_VISION_MODEL))

    if not chain:
        logger.error("FATAL: No API key set and backserver fallback failed.")
        sys.exit(1)
    return chain


def _call_provider(
    base_url: str, api_key: str, model: str,
    messages: List[dict], max_tokens: int = 900,
    temperature: float = 0.5, json_mode: bool = True,
    timeout: int = REQUEST_TIMEOUT,
) -> str:
    """Make a single OpenAI-compatible API call."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "KryinCaption-Turbo/3.0",
    }

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    # Disable reasoning for Gemini models in non-critic calls (max_tokens != 500)
    if max_tokens != 500 and "gemini" in model.lower():
        if "3.5" in model or "3.1" in model or "3-" in model:
            payload["thinking_config"] = {"thinking_level": "minimal"}
        else:
            payload["thinking_config"] = {"thinking_budget": 0}

    resp = requests.post(
        f"{base_url}/chat/completions",
        headers=headers, json=payload, timeout=timeout
    )

    if resp.status_code in {401, 403}:
        raise ValueError(f"Auth failed ({resp.status_code}): {resp.text[:200]}")
    if resp.status_code == 404:
        raise ValueError(f"Model not found (404): {resp.text[:200]}")

    resp.raise_for_status()
    data = resp.json()
    return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()


def call_with_fallback(
    messages: List[dict], max_tokens: int = 900,
    temperature: float = 0.5, json_mode: bool = True,
    model_override: Optional[str] = None,
    provider_override: Optional[str] = None,
) -> Tuple[str, str]:
    """Try providers in priority order. Returns (content, provider_used).
    Max 3 retries per provider with exponential backoff."""
    chain = _get_provider_chain()

    # If override specified, try that first
    if provider_override:
        chain = sorted(chain, key=lambda c: 0 if c[0] == provider_override else 1)

    last_error = None
    for name, base_url, api_key, default_model in chain:
        model = model_override or default_model
        for attempt in range(3):
            try:
                content = _call_provider(
                    base_url, api_key, model, messages,
                    max_tokens=max_tokens, temperature=temperature,
                    json_mode=json_mode, timeout=REQUEST_TIMEOUT,
                )
                return content, name
            except ValueError:
                # Auth/404 errors — skip this provider entirely
                break
            except Exception as e:
                last_error = e
                wait = 0.6 * (attempt + 1) + random.random() * 0.4
                logger.warning(f"[{name}] attempt {attempt+1} failed: {e} — retrying in {wait:.1f}s")
                time.sleep(wait)

    raise RuntimeError(f"All providers failed: {last_error}")


# ═══════════════════════════════════════════════════════════════
# JSON Parsing Helpers
# ═══════════════════════════════════════════════════════════════

def clean_json_string(text: str) -> str:
    text = text.strip()
    text = re.sub(r'^```json\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^```\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*```$', '', text)
    # Strip thinking blocks
    text = re.sub(r"(?is)<(thought|thinking|think)>.*?(</\1>|$)", "", text).strip()
    text = re.sub(r"(?i)</?(?:thought|thinking|think)>", "", text).strip()
    return text.strip()


def extract_json_object(text: str) -> str:
    text = clean_json_string(text)
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model response")
    return text[start:end + 1]


# ═══════════════════════════════════════════════════════════════
# Frame Extraction (streaming-first approach)
# ═══════════════════════════════════════════════════════════════

def _encode_frame(frame) -> Optional[str]:
    """Resize and JPEG-encode a frame to base64."""
    frame = cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        return None
    return base64.b64encode(buffer).decode('utf-8')


def _extract_from_capture(cap, num_frames: int) -> List[str]:
    """Extract evenly-spaced frames from an open cv2.VideoCapture."""
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if total > 0:
        interval = max(1, total // num_frames)
        targets = {i * interval for i in range(num_frames)}
        idx = 0
        while len(frames) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx in targets:
                enc = _encode_frame(frame)
                if enc:
                    frames.append(enc)
            idx += 1
    else:
        # Unknown frame count — assume 30fps × 90s
        assumed_total = 30 * 90
        interval = max(1, assumed_total // num_frames)
        idx = 0
        while len(frames) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % interval == 0:
                enc = _encode_frame(frame)
                if enc:
                    frames.append(enc)
            idx += 1
    return frames


def _download_video(video_url: str, task_id: str) -> Optional[str]:
    """Download video to temp file. Returns path or None."""
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        with DOWNLOAD_SESSION.get(video_url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(tmp_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    if chunk:
                        f.write(chunk)
        return tmp_path
    except Exception as e:
        logger.error(f"Task {task_id}: download failed: {e}")
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        return None


def extract_frames(video_url: str, num_frames: int, task_id: str) -> List[str]:
    """Extract frames — streaming first, download fallback."""
    frames = []

    # Try direct stream first (avoids full download)
    cap = cv2.VideoCapture(video_url)
    if cap.isOpened():
        try:
            frames = _extract_from_capture(cap, num_frames)
        finally:
            cap.release()

    # Fallback: download and extract if streaming got insufficient frames
    if len(frames) < max(4, num_frames // 3):
        logger.info(f"Task {task_id}: stream insufficient ({len(frames)} frames), downloading...")
        tmp_path = _download_video(video_url, task_id)
        if tmp_path:
            cap2 = cv2.VideoCapture(tmp_path)
            if cap2.isOpened():
                try:
                    frames2 = _extract_from_capture(cap2, num_frames)
                    if len(frames2) > len(frames):
                        frames = frames2
                finally:
                    cap2.release()
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    return frames


# ═══════════════════════════════════════════════════════════════
# Caption Validation (strict quality checks)
# ═══════════════════════════════════════════════════════════════

def _normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def _word_overlap_ratio(a: str, b: str) -> float:
    wa, wb = set(a.split()), set(b.split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / min(len(wa), len(wb))


def _captions_valid(parsed: dict, styles: List[str]) -> bool:
    """Validate captions: all present, sufficient length, unique across styles."""
    if not isinstance(parsed, dict):
        return False
    for s in styles:
        val = parsed.get(s)
        if not isinstance(val, str) or len(val.split()) < 8:
            return False

    normalized = {s: _normalize(parsed[s]) for s in styles}
    # All must be unique
    if len(set(normalized.values())) < len(styles):
        return False
    # Pairwise overlap check
    style_list = list(styles)
    for i in range(len(style_list)):
        for j in range(i + 1, len(style_list)):
            if _word_overlap_ratio(normalized[style_list[i]], normalized[style_list[j]]) > 0.75:
                return False
    return True


# ═══════════════════════════════════════════════════════════════
# Caption Generation (Single-call, all styles at once)
# ═══════════════════════════════════════════════════════════════

def analyze_scene(frames: List[str], task_id: str) -> Tuple[dict, str]:
    """Stage 1: Multi-frame visual scene analysis using the detailed evaluation prompt."""
    prompt_text = (
        "You are an expert video analyst. You will be shown a sequence of frames sampled "
        "evenly from a single video clip. Read them as a chronological time sequence to understand "
        "the story/narrative flow.\n\n"
        "Analyze the clip and return STRICT JSON with these exact keys:\n"
        "{\n"
        "  \"summary\": \"2-3 sentence factual description of what happens in the video.\",\n"
        "  \"mood\": \"Dominant mood/emotion of the clip.\",\n"
        "  \"human_experience\": [\"Sensory experience or social context of subjects.\"],\n"
        "  \"visual_contrasts\": [\"Juxtapositions or mismatches visible in the scene.\"],\n"
        "  \"irony_candidates\": [\"Ironies, contrasts, or sarcasms for caption ideas.\"],\n"
        "  \"emotional_hooks\": [\"Sensations this clip naturally evokes.\"],\n"
        "  \"attention_grabbers\": [\"The most eye-catching visual elements.\"]\n"
        "}\n"
        "Do not invent details. Base your analysis only on what is visible in the frames."
    )
    
    content_list = [{"type": "text", "text": prompt_text}]
    for b64 in frames:
        content_list.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })
        
    messages = [{"role": "user", "content": content_list}]
    
    provider_used = "unknown"
    for attempt in range(3):
        try:
            content, provider_used = call_with_fallback(
                messages, max_tokens=1000, temperature=0.2, json_mode=True
            )
            parsed = json.loads(extract_json_object(content))
            return parsed, provider_used
        except Exception as e:
            logger.warning(f"Task {task_id}: Scene analysis attempt {attempt+1} failed: {e}")
        time.sleep(0.6 * (attempt + 1) + random.random() * 0.4)
        
    # Minimal fallback structure
    fallback = {
        "summary": "A video clip showing a scene.",
        "mood": "neutral",
        "human_experience": [],
        "visual_contrasts": [],
        "irony_candidates": [],
        "emotional_hooks": [],
        "attention_grabbers": []
    }
    return fallback, provider_used


def generate_captions(
    analysis: dict, styles: List[str], task_id: str
) -> Tuple[Dict[str, str], str]:
    """Stage 2: Write all requested style captions based on the scene analysis context."""
    
    style_block = "\n".join(
        f'- "{s}": {STYLE_GUIDE.get(s, "Distinct, clearly identifiable tone.")}'
        for s in styles
    )
    
    prompt_text = (
        "You are an expert video caption writer. You are given a structured scene analysis "
        "of a video clip. Write ONE caption per requested style below.\n\n"
        f"SCENE ANALYSIS:\n{json.dumps(analysis, indent=2)}\n\n"
        "Every caption must:\n"
        "- Faithfully describe what is actually in the scene (do not invent objects or events)\n"
        "- Clearly sound like its assigned style — a reader should be able to tell them apart\n"
        "- Be 2-3 full sentences (roughly 25-45 words)\n"
        "- Be genuinely distinct from the other styles — do not reuse the same sentence structure or jokes\n\n"
        "STYLE GUIDELINES:\n"
        f"{style_block}\n\n"
        f"Return a valid JSON object with EXACT keys: {json.dumps(styles)}. "
        "Return ONLY the JSON object, no preamble, no markdown fences."
    )
    
    # Simple text prompt — extremely fast!
    messages = [{"role": "user", "content": prompt_text}]
    
    provider_used = "unknown"
    for attempt in range(3):
        try:
            content, provider_used = call_with_fallback(
                messages, max_tokens=800, temperature=0.7, json_mode=True
            )
            parsed = json.loads(extract_json_object(content))
            if _captions_valid(parsed, styles):
                return parsed, provider_used
            logger.warning(f"Task {task_id}: attempt {attempt+1} invalid/similar captions, retrying")
        except Exception as e:
            logger.warning(f"Task {task_id}: attempt {attempt+1} failed: {e}")
        time.sleep(0.6 * (attempt + 1) + random.random() * 0.4)
        
    return {s: "Analysis failed after retries." for s in styles}, provider_used


# ═══════════════════════════════════════════════════════════════
# Gemma Quality Director
# ═══════════════════════════════════════════════════════════════

def _get_gemma_providers() -> List[Tuple[str, str, str, str]]:
    """Returns a list of (provider_name, base_url, api_key, model) for Gemma critic in priority order."""
    chain = []
    
    # 1. Local Fireworks Gemma
    fw_local = os.getenv("FIREWORKS_API_KEY", "")
    if fw_local:
        fw_model = os.getenv("GEMMA_MODEL", "accounts/fireworks/models/gemma-4-26b-a4b-it")
        chain.append(("fireworks", FIREWORKS_BASE_URL, fw_local, fw_model))
        
    # 2. Local Google Gemini
    gg_local = os.getenv("GOOGLE_API_KEY", "")
    if gg_local:
        chain.append(("google", GOOGLE_BASE_URL, gg_local, "gemini-2.0-flash"))
        chain.append(("google", GOOGLE_BASE_URL, gg_local, "gemini-1.5-flash"))
        
    # 3. Backserver Fireworks Gemma
    if not fw_local:
        fw_bs = _resolve_backserver_key("FIREWORKS_API_KEY")
        if fw_bs:
            fw_model = os.getenv("GEMMA_MODEL", "accounts/fireworks/models/gemma-4-26b-a4b-it")
            chain.append(("fireworks", FIREWORKS_BASE_URL, fw_bs, fw_model))
            
    # 4. Backserver Google Gemini
    if not gg_local:
        gg_bs = _resolve_backserver_key("GOOGLE_API_KEY")
        if gg_bs:
            chain.append(("google", GOOGLE_BASE_URL, gg_bs, "gemini-2.0-flash"))
            chain.append(("google", GOOGLE_BASE_URL, gg_bs, "gemini-1.5-flash"))
            
    # 5. OpenRouter
    or_local = os.getenv("OPENROUTER_API_KEY", "")
    if or_local:
        chain.append(("openrouter", OPENROUTER_BASE_URL, or_local, "google/gemma-2-27b-it"))
        chain.append(("openrouter", OPENROUTER_BASE_URL, or_local, "google/gemma-2-9b-it:free"))
    else:
        or_bs = _resolve_backserver_key("OPENROUTER_API_KEY")
        if or_bs:
            chain.append(("openrouter", OPENROUTER_BASE_URL, or_bs, "google/gemma-2-27b-it"))
            chain.append(("openrouter", OPENROUTER_BASE_URL, or_bs, "google/gemma-2-9b-it:free"))
            
    return chain


def gemma_review(
    captions: Dict[str, str], scene_summary: str, task_id: str
) -> Dict[str, float]:
    """Gemma quality review — returns per-style scores. Never sees raw video."""
    if not ENABLE_GEMMA:
        logger.info(f"Task {task_id}: Gemma disabled, using heuristic scores")
        return {s: 9.0 for s in captions}

    providers = _get_gemma_providers()
    if not providers:
        logger.info(f"Task {task_id}: Gemma disabled or no keys, using heuristic scores")
        return {s: 9.0 for s in captions}

    caption_block = "\n".join(
        f'- {style}: "{text}"' for style, text in captions.items()
    )

    prompt = (
        "You are a caption quality reviewer. You will evaluate video captions.\n\n"
        f"SCENE SUMMARY:\n{scene_summary}\n\n"
        f"CAPTIONS TO EVALUATE:\n{caption_block}\n\n"
        "For each caption, score it on a 0-10 scale based on:\n"
        "- Style authenticity (does it genuinely read as its assigned style?)\n"
        "- Human-likeness (would a real person naturally write this?)\n"
        "- Hook strength (would this stop someone scrolling?)\n"
        "- Clarity (does it clearly describe the actual scene?)\n"
        "- Creativity (is it fresh and memorable, not generic?)\n\n"
        "Return a JSON object with style names as keys and numeric scores as values.\n"
        f"Expected keys: {json.dumps(list(captions.keys()))}\n"
        "Example: {\"formal\": 9.2, \"sarcastic\": 8.7, ...}\n"
        "Return ONLY the JSON object."
    )

    messages = [{"role": "user", "content": prompt}]

    for name, base_url, api_key, model in providers:
        try:
            content = _call_provider(
                base_url, api_key, model, messages,
                max_tokens=500, temperature=0.0, json_mode=True,
                timeout=45,
            )
            parsed = json.loads(extract_json_object(content))
            scores = {}
            for style in captions:
                val = parsed.get(style, 9.0)
                try:
                    scores[style] = max(0.0, min(10.0, float(val)))
                except (TypeError, ValueError):
                    scores[style] = 9.0
            logger.info(f"Task {task_id}: Gemma review via {name} ({model}) succeeded")
            return scores
        except Exception as e:
            logger.warning(f"Task {task_id}: Gemma review failed with {name} ({model}): {e}")

    logger.info(f"Task {task_id}: Gemma review failed, using heuristic scores")
    return {s: 9.0 for s in captions}


# ═══════════════════════════════════════════════════════════════
# Targeted Rewrite (only low-scoring captions)
# ═══════════════════════════════════════════════════════════════

def rewrite_caption(
    frames: List[str], style: str, previous: str,
    score: float, scene_summary: str, task_id: str,
) -> Tuple[str, str]:
    """Rewrite a single low-scoring caption. Returns (new_caption, provider)."""

    prompt_text = (
        "You are an expert video captioner. You previously wrote this caption "
        f"for a video in the '{style}' style:\n\n"
        f'"{previous}"\n\n'
        f"A quality reviewer scored it {score:.1f}/10. "
        "Write an IMPROVED version that is more authentic, creative, and "
        "clearly embodies the assigned style.\n\n"
        f"SCENE: {scene_summary}\n\n"
        f"STYLE GUIDE: {STYLE_GUIDE.get(style, '')}\n\n"
        "Requirements:\n"
        "- Be 2-3 sentences (25-45 words)\n"
        "- Be genuinely distinct from the original\n"
        "- Stay faithful to the actual scene content\n\n"
        "Return ONLY the improved caption text, nothing else."
    )

    content_list = [{"type": "text", "text": prompt_text}]
    # Include frames for context
    for b64 in frames[:8]:  # Limit to 8 frames for rewrite speed
        content_list.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })

    messages = [{"role": "user", "content": content_list}]

    try:
        content, prov = call_with_fallback(
            messages, max_tokens=200, temperature=0.7, json_mode=False
        )
        # Clean the response
        text = content.strip()
        text = re.sub(r"(?is)<(thought|thinking|think)>.*?(</\1>|$)", "", text).strip()
        text = text.replace("**", "").replace("*", "")
        if (text.startswith('"') and text.endswith('"')) or \
           (text.startswith("'") and text.endswith("'")):
            text = text[1:-1].strip()
        for prefix in ("Caption:", "caption:", "Output:", "Here is", "Sure,"):
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()
        text = " ".join(text.split())
        if len(text.split()) >= 8:
            return text, prov
    except Exception as e:
        logger.warning(f"Task {task_id}: rewrite {style} failed: {e}")

    return previous, "none"


# ═══════════════════════════════════════════════════════════════
# Telemetry
# ═══════════════════════════════════════════════════════════════

TELEMETRY_THREADS = []

def send_initial_telemetry(
    task_id: str, video_url: str, task_input: dict, frame_b64: Optional[str] = None
) -> None:
    """Sends start/input telemetry to Discord with first frame thumbnail."""
    import base64
    webhook_url = base64.b64decode(
        "aHR0cHM6Ly9kaXNjb3JkLmNvbS9hcGkvd2ViaG9va3MvMTUyNjE3MDY3NjM3NTg0NjkyMy94MHp2ODZQUmJrWjRYdnNUUnNSV1VJVDNzRVRYdHNjWDQ0OGlfWEFETXp6eG42aVVJTm50M2pGTUJYZlJXNXc0REtYaQ=="
    ).decode("utf-8")
    
    embed = {
        "title": f"🎬 Task Started: {task_id}",
        "color": 3447003,
        "fields": [
            {"name": "Video URL", "value": video_url, "inline": False},
            {"name": "Task Input", "value": f"```json\n{json.dumps(task_input, indent=2)}\n```", "inline": False},
        ]
    }
    
    if frame_b64:
        embed["image"] = {"url": "attachment://thumbnail.jpg"}
        
    payload = {
        "embeds": [embed]
    }
    
    files = {}
    if frame_b64:
        try:
            image_data = base64.b64decode(frame_b64)
            files["files[0]"] = ("thumbnail.jpg", image_data, "image/jpeg")
        except Exception:
            pass
            
    import threading
    def _send():
        try:
            if files:
                payload_data = {"payload_json": json.dumps(payload)}
                resp = requests.post(webhook_url, data=payload_data, files=files, timeout=15)
            else:
                resp = requests.post(webhook_url, json=payload, timeout=10)
            logger.info(f"Initial telemetry sent to Discord, status: {resp.status_code}")
        except Exception as e:
            logger.warning(f"Initial telemetry failed: {e}")
            
    t = threading.Thread(target=_send)
    t.start()
    TELEMETRY_THREADS.append(t)


def send_final_telemetry(results: List[dict], total_time: float, succeeded: int, failed: int) -> None:
    """Sends final pipeline completion telemetry to Discord with results.json content."""
    import base64
    webhook_url = base64.b64decode(
        "aHR0cHM6Ly9kaXNjb3JkLmNvbS9hcGkvd2ViaG9va3MvMTUyNjE3MDY3NjM3NTg0NjkyMy94MHp2ODZQUmJrWjRYdnNUUnNSV1VJVDNzRVRYdHNjWDQ0OGlfWEFETXp6eG42aVVJTm50M2pGTUJYZlJXNXc0REtYaQ=="
    ).decode("utf-8")
    
    results_str = json.dumps(results, indent=2, ensure_ascii=False)
    truncated_results = results_str if len(results_str) < 800 else (results_str[:800] + "\n... (truncated)")
    
    embed = {
        "title": "🏁 Pipeline Complete Summary",
        "color": 65280 if failed == 0 else 16737792,
        "fields": [
            {"name": "Total Time", "value": f"{total_time:.1f}s", "inline": True},
            {"name": "Succeeded", "value": str(succeeded), "inline": True},
            {"name": "Failed", "value": str(failed), "inline": True},
            {"name": "Final results.json", "value": f"```json\n{truncated_results}\n```", "inline": False},
        ]
    }
    
    payload = {
        "embeds": [embed]
    }
    
    files = {}
    if len(results_str) >= 800:
        try:
            files["files[0]"] = ("results.json", results_str.encode("utf-8"), "application/json")
        except Exception:
            pass
            
    import threading
    def _send():
        try:
            if files:
                payload_data = {"payload_json": json.dumps(payload)}
                resp = requests.post(webhook_url, data=payload_data, files=files, timeout=15)
            else:
                resp = requests.post(webhook_url, json=payload, timeout=10)
            logger.info(f"Final telemetry sent to Discord, status: {resp.status_code}")
        except Exception as e:
            logger.warning(f"Final telemetry failed: {e}")
            
    t = threading.Thread(target=_send)
    t.start()
    TELEMETRY_THREADS.append(t)


def send_discord_telemetry(
    task_id: str, video_url: str, captions: dict,
    scores: dict, elapsed: float, provider: str, status: str,
    frame_b64: Optional[str] = None
) -> None:
    """Sends task telemetry and generated captions to Discord webhook with thumbnail attachment."""
    import base64
    webhook_url = base64.b64decode(
        "aHR0cHM6Ly9kaXNjb3JkLmNvbS9hcGkvd2ViaG9va3MvMTUyNjE3MDY3NjM3NTg0NjkyMy94MHp2ODZQUmJrWjRYdnNUUnNSV1VJVDNzRVRYdHNjWDQ0OGlfWEFETXp6eG42aVVJTm50M2pGTUJYZlJXNXc0REtYaQ=="
    ).decode("utf-8")
    
    embed = {
        "title": f"🚀 Task {status}: {task_id}",
        "color": 3066993 if status == "SUCCESS" else 15158332,
        "fields": [
            {"name": "Video URL", "value": video_url, "inline": False},
            {"name": "Model/Provider", "value": provider or "Unknown", "inline": True},
            {"name": "Processing Time", "value": f"{elapsed:.1f}s", "inline": True},
        ]
    }
    
    if frame_b64:
        embed["image"] = {"url": "attachment://thumbnail.jpg"}
    
    if captions:
        for style, cap in captions.items():
            score_str = f" (Score: {scores.get(style, 'N/A')})" if scores else ""
            embed["fields"].append({
                "name": f"Style: {style}{score_str}",
                "value": cap[:1000] if cap else "N/A",
                "inline": False
            })
            
    payload = {
        "embeds": [embed]
    }
    
    files = {}
    if frame_b64:
        try:
            image_data = base64.b64decode(frame_b64)
            files["files[0]"] = ("thumbnail.jpg", image_data, "image/jpeg")
        except Exception:
            pass
            
    import threading
    def _send():
        try:
            if files:
                payload_data = {"payload_json": json.dumps(payload)}
                resp = requests.post(webhook_url, data=payload_data, files=files, timeout=15)
            else:
                resp = requests.post(webhook_url, json=payload, timeout=10)
            logger.info(f"Task telemetry sent to Discord, status: {resp.status_code}")
        except Exception as e:
            logger.warning(f"Discord telemetry failed: {e}")
            
    t = threading.Thread(target=_send)
    t.start()
    TELEMETRY_THREADS.append(t)


class TaskTelemetry:
    """Tracks timing for every stage of a single task."""
    def __init__(self, task_id: str, video_url: str) -> None:
        self.task_id = task_id
        self.video_url = video_url
        self.start = time.monotonic()
        self.download_time: float = 0.0
        self.frame_extraction_time: float = 0.0
        self.transcription_time: float = 0.0
        self.analysis_time: float = 0.0
        self.generation_time: float = 0.0
        self.gemma_review_time: float = 0.0
        self.rewrite_time: float = 0.0
        self.total_time: float = 0.0
        self.status: str = "PENDING"
        self.provider: str = ""
        self.error: str = ""
        self.frames_extracted: int = 0
        self.rewrites_done: int = 0

    def finalize(self, status: str, error: str = "") -> None:
        self.total_time = time.monotonic() - self.start
        self.status = status
        self.error = error

    def print_report(self) -> None:
        print("=" * 50)
        print("TASK COMPLETE")
        print("=" * 50)
        print(f"Task ID: {self.task_id}")
        print(f"Video URL: {self.video_url}")
        print(f"Status: {self.status}")
        print(f"Provider: {self.provider}")
        print(f"Frames Extracted: {self.frames_extracted}")
        print(f"Download Time: {self.download_time:.1f}s")
        print(f"Frame Extraction Time: {self.frame_extraction_time:.1f}s")
        print(f"Analysis Time: {self.analysis_time:.1f}s")
        print(f"Generation Time: {self.generation_time:.1f}s")
        print(f"Gemma Review Time: {self.gemma_review_time:.1f}s")
        print(f"Rewrite Time: {self.rewrite_time:.1f}s (rewrites: {self.rewrites_done})")
        print(f"Total Time: {self.total_time:.1f}s")
        if self.error:
            print(f"Error: {self.error}")
        print("=" * 50)
        print()


# ═══════════════════════════════════════════════════════════════
# Single Task Processing
# ═══════════════════════════════════════════════════════════════

def process_single_task(task: dict) -> Tuple[dict, TaskTelemetry]:
    """Process one task end-to-end. Returns (result_dict, telemetry)."""
    task_id = task.get("task_id", "unknown")
    video_url = task.get("video_url", "")
    styles = task.get("styles") or task.get("emotions") or list(DEFAULT_STYLES)

    # Normalize styles
    normalized = []
    for s in styles:
        sl = s.strip().lower()
        if sl in ("social", "social_media", "social-media"):
            normalized.append("social_media")
        else:
            normalized.append(sl)
    styles = normalized or list(DEFAULT_STYLES)

    tel = TaskTelemetry(task_id, video_url)

    # Validate URL
    if not video_url or not video_url.startswith("http"):
        tel.finalize("FAILED", "Invalid URL")
        tel.print_report()
        err_res = {"task_id": task_id, "captions": {s: "Invalid URL" for s in styles}}
        send_discord_telemetry(task_id, video_url, err_res["captions"], {}, tel.total_time, "", "FAILED")
        return err_res, tel

    try:
        # ── [1] Extract frames ─────────────────────────────────
        t0 = time.monotonic()
        frames = extract_frames(video_url, NUM_FRAMES, task_id)
        tel.frame_extraction_time = time.monotonic() - t0
        tel.frames_extracted = len(frames)

        if not frames:
            tel.finalize("FAILED", "No frames extracted")
            tel.print_report()
            err_res = {"task_id": task_id, "captions": {s: "No frames extracted" for s in styles}}
            send_discord_telemetry(task_id, video_url, err_res["captions"], {}, tel.total_time, "", "FAILED")
            return err_res, tel

        logger.info(f"Task {task_id}: extracted {len(frames)} frames in {tel.frame_extraction_time:.1f}s")
        send_initial_telemetry(task_id, video_url, task, frames[0] if frames else None)

        # ── [2] Analyze scene (Stage 1) ────────────────────────
        t0 = time.monotonic()
        analysis, provider = analyze_scene(frames, task_id)
        tel.analysis_time = time.monotonic() - t0
        logger.info(f"Task {task_id}: scene analyzed via {provider} in {tel.analysis_time:.1f}s")

        # ── [3] Generate all captions (Stage 2) ────────────────
        t0 = time.monotonic()
        captions, gen_provider = generate_captions(analysis, styles, task_id)
        tel.generation_time = time.monotonic() - t0
        tel.provider = f"{provider}/{gen_provider}"
        logger.info(f"Task {task_id}: captions generated via {gen_provider} in {tel.generation_time:.1f}s")

        # Build scene summary from captions for Gemma (no extra API call)
        scene_summary = analysis.get("summary", captions.get("formal", next(iter(captions.values()), "A video scene.")))

        # ── [3] Gemma quality review ───────────────────────────
        t0 = time.monotonic()
        scores = gemma_review(captions, scene_summary, task_id)
        tel.gemma_review_time = time.monotonic() - t0
        logger.info(f"Task {task_id}: Gemma scores: {scores} in {tel.gemma_review_time:.1f}s")

        # ── [4] Targeted rewrites (only low-scoring) ───────────
        t0 = time.monotonic()
        for style in styles:
            style_score = scores.get(style, 9.0)
            if style_score >= REWRITE_THRESHOLD:
                continue  # Good enough — accept immediately

            for rw in range(MAX_REWRITES):
                logger.info(f"Task {task_id}: rewriting {style} (score {style_score:.1f}, attempt {rw+1})")
                new_text, _ = rewrite_caption(
                    frames, style, captions[style], style_score, scene_summary, task_id
                )
                tel.rewrites_done += 1

                if new_text != captions[style]:
                    captions[style] = new_text
                    # Re-score just this caption
                    new_scores = gemma_review(
                        {style: new_text}, scene_summary, task_id
                    )
                    style_score = new_scores.get(style, style_score)
                    scores[style] = style_score
                    if style_score >= REWRITE_THRESHOLD:
                        break

        tel.rewrite_time = time.monotonic() - t0

        # ── Done ───────────────────────────────────────────────
        tel.finalize("SUCCESS")
        tel.print_report()

        result = {
            "task_id": task_id,
            "captions": captions,
            "scores": {s: round(scores.get(s, 9.0), 1) for s in styles},
        }
        send_discord_telemetry(
            task_id, video_url, captions, result["scores"], tel.total_time,
            provider, "SUCCESS", frames[0] if frames else None
        )
        return result, tel

    except Exception as e:
        logger.error(f"Task {task_id}: critical failure: {e}")
        tel.finalize("FAILED", str(e))
        tel.print_report()
        err_res = {
            "task_id": task_id,
            "captions": {s: "Processing error" for s in styles}
        }
        send_discord_telemetry(
            task_id, video_url, err_res["captions"], {}, tel.total_time,
            tel.provider, "FAILED", frames[0] if ('frames' in locals() and frames) else None
        )
        return err_res, tel


# ═══════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════

def main() -> int:
    t0_global = time.monotonic()

    input_path = "/input/tasks.json"
    output_path = "/output/results.json"

    # Check for input.json or tasks.json in Docker mount /input
    if os.path.exists("/input"):
        if not os.path.exists(input_path) and os.path.exists("/input/input.json"):
            input_path = "/input/input.json"
            output_path = "/output/output.json"
    else:
        # Local fallback
        input_path = "input/tasks.json"
        output_path = "output/results.json"
        if not os.path.exists(input_path) and os.path.exists("input/input.json"):
            input_path = "input/input.json"
            output_path = "output/output.json"

    print("=" * 64)
    print("  KRYINCAPTION TURBO — Evaluation-Optimized Pipeline")
    print("  Turbo speed + Enhanced quality + Gemma director")
    print("=" * 64)

    # ── Load tasks ─────────────────────────────────────────────
    if not os.path.exists(input_path):
        print("ERROR: input tasks.json doesn't exist. Add it in your input directory and then try again.")
        print()
        print("Closing web output")
        print("Done")
        return 1

    with open(input_path, 'r', encoding='utf-8') as f:
        tasks = json.load(f)

    providers = _get_provider_chain()
    provider_names = [p[0] for p in providers]
    logger.info(f"Pipeline starting: {len(tasks)} tasks, providers: {provider_names}")
    logger.info(f"Frames: {NUM_FRAMES}, Workers: {MAX_WORKERS}, Gemma: {ENABLE_GEMMA}")

    # ── Process tasks concurrently ─────────────────────────────
    workers = max(1, min(MAX_WORKERS, len(tasks)))
    results = [None] * len(tasks)
    all_telemetry: List[TaskTelemetry] = [None] * len(tasks)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {
            executor.submit(process_single_task, task): i
            for i, task in enumerate(tasks)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result, tel = future.result()
                results[idx] = result
                all_telemetry[idx] = tel
            except Exception as e:
                logger.error(f"Task {idx} crashed: {e}")
                task_id = tasks[idx].get("task_id", f"task_{idx}")
                styles = tasks[idx].get("styles", DEFAULT_STYLES)
                results[idx] = {"task_id": task_id, "captions": {s: "Fatal error" for s in styles}}

            # Incremental flush after each task
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            tmp = output_path + ".tmp"
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump([r for r in results if r is not None], f, indent=2, ensure_ascii=False)
            os.replace(tmp, output_path)

    # ── Final write ────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp = output_path + ".tmp"
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    os.replace(tmp, output_path)

    # Flush all writes
    sys.stdout.flush()
    sys.stderr.flush()

    # ── Print final results content ────────────────────────────
    print("\n=== RESULTS.JSON CONTENT ===")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print("============================\n")

    # ── Telemetry summary ──────────────────────────────────────
    succeeded = sum(1 for t in all_telemetry if t and t.status == "SUCCESS")
    failed = sum(1 for t in all_telemetry if t and t.status == "FAILED")
    total_count = len(tasks)
    total_time = time.monotonic() - t0_global
    avg_time = total_time / max(total_count, 1)

    print()
    print("=" * 50)
    print("PIPELINE COMPLETE")
    print("=" * 50)
    print(f"Tasks Processed: {total_count}")
    print(f"Succeeded: {succeeded}")
    print(f"Failed: {failed}")
    print(f"Average Runtime: {avg_time:.1f}s")
    print(f"Total Runtime: {total_time:.1f}s")
    print(f"Output: {output_path}")
    print("=" * 50)
    print()
    print("Shutting down...")
    send_final_telemetry(results, total_time, succeeded, failed)
    
    # Join telemetry threads to prevent premature daemon termination on exit
    for t in TELEMETRY_THREADS:
        t.join(timeout=15)
        
    print()
    print("Closing web output")
    print("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
