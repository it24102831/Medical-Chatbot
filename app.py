import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from pinecone import Pinecone
from requests import Response
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.prompt import system_prompt

load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_REQUEST_BYTES", "16384"))

DEFAULT_INDEX_NAME = "medical-chatbot-v2"
DEFAULT_NAMESPACE = "medical-book"
DEFAULT_MODEL = "openai/gpt-4o-mini"
DEFAULT_EMBED_MODEL = "llama-text-embed-v2"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

_pinecone_client: Optional[Pinecone] = None
_pinecone_index = None
_http_session: Optional[requests.Session] = None
_client_lock = threading.Lock()


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name, default)
    return value.strip() if isinstance(value, str) else value


def _is_placeholder(value: Optional[str]) -> bool:
    if not value:
        return True
    lowered = value.lower()
    return "your_" in lowered or lowered in {"changeme", "replace-me", "example"}


def _required_config_status() -> Dict[str, bool]:
    return {
        "PINECONE_API_KEY": not _is_placeholder(_env("PINECONE_API_KEY")),
        "OPENROUTER_API_KEY": not _is_placeholder(_env("OPENROUTER_API_KEY")),
        "PINECONE_INDEX_NAME": bool(_env("PINECONE_INDEX_NAME", DEFAULT_INDEX_NAME)),
        "PINECONE_NAMESPACE": bool(_env("PINECONE_NAMESPACE", DEFAULT_NAMESPACE)),
    }


def _build_http_session() -> requests.Session:
    retry = Retry(
        total=1,
        connect=1,
        read=1,
        backoff_factor=0.3,
        status_forcelist=(502, 503, 504),
        allowed_methods=frozenset({"POST", "GET"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=8)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def get_http_session() -> requests.Session:
    global _http_session
    if _http_session is None:
        with _client_lock:
            if _http_session is None:
                _http_session = _build_http_session()
    return _http_session


def get_pinecone_client() -> Pinecone:
    global _pinecone_client
    api_key = _env("PINECONE_API_KEY")
    if _is_placeholder(api_key):
        raise ValueError("Pinecone API key is not configured.")

    if _pinecone_client is None:
        with _client_lock:
            if _pinecone_client is None:
                _pinecone_client = Pinecone(api_key=api_key)
    return _pinecone_client


def get_pinecone_index():
    global _pinecone_index
    if _pinecone_index is None:
        with _client_lock:
            if _pinecone_index is None:
                client = get_pinecone_client()
                index_name = _env("PINECONE_INDEX_NAME", DEFAULT_INDEX_NAME)
                if not index_name:
                    raise ValueError("Pinecone index name is missing.")
                try:
                    _pinecone_index = client.Index(index_name)
                except Exception as exc:
                    logger.exception("Failed to initialize Pinecone index client")
                    raise RuntimeError("Unable to initialize Pinecone index client.") from exc
    return _pinecone_index


def _normalize_hits(search_response: Any) -> List[Dict[str, Any]]:
    if search_response is None:
        return []

    if isinstance(search_response, dict):
        result = search_response.get("result", {})
        hits = result.get("hits") if isinstance(result, dict) else search_response.get("hits")
        return hits if isinstance(hits, list) else []

    result = getattr(search_response, "result", None)
    if result is not None:
        hits = getattr(result, "hits", None)
        if isinstance(hits, list):
            return hits

    hits = getattr(search_response, "hits", None)
    return hits if isinstance(hits, list) else []


def _extract_hit_field(hit: Dict[str, Any], key: str) -> Any:
    fields = hit.get("fields") if isinstance(hit, dict) else None
    if isinstance(fields, dict) and key in fields:
        return fields.get(key)
    return hit.get(key) if isinstance(hit, dict) else None


def retrieve_context(question: str) -> str:
    namespace = _env("PINECONE_NAMESPACE", DEFAULT_NAMESPACE)
    top_k = max(1, min(int(_env("TOP_K", "3") or "3"), 10))
    max_context_chars = max(1000, int(_env("MAX_CONTEXT_CHARS", "12000") or "12000"))
    per_chunk_chars = max(200, int(_env("MAX_CHUNK_CHARS", "1800") or "1800"))

    index = get_pinecone_index()
    try:
        response = index.search_records(
            namespace=namespace,
            query={"inputs": {"text": question}, "top_k": top_k},
            fields=["text", "source", "filename", "page", "chunk_number"],
        )
    except Exception as exc:
        msg = str(exc).lower()
        if "unauthorized" in msg or "authentication" in msg:
            raise PermissionError("Pinecone authentication failed.") from exc
        if "timeout" in msg:
            raise TimeoutError("Pinecone request timed out.") from exc
        if "rate" in msg or "too many" in msg:
            raise RuntimeError("Pinecone rate limit reached.") from exc
        raise ConnectionError("Pinecone query failed.") from exc

    hits = _normalize_hits(response)
    if not hits:
        return ""

    snippets: List[str] = []
    total = 0
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        text = _extract_hit_field(hit, "text")
        if not isinstance(text, str):
            continue
        text = text.strip().replace("\x00", "")
        if not text:
            continue
        text = text[:per_chunk_chars]
        source = _extract_hit_field(hit, "source") or _extract_hit_field(hit, "filename") or "unknown"
        page = _extract_hit_field(hit, "page")
        header = f"Source: {source}" + (f" | Page: {page}" if page is not None else "")
        block = f"{header}\n{text}"
        if total + len(block) + 2 > max_context_chars:
            break
        snippets.append(block)
        total += len(block) + 2

    return "\n\n".join(snippets)


def _openrouter_chat_completion(messages: List[Dict[str, str]]) -> Response:
    api_key = _env("OPENROUTER_API_KEY")
    if _is_placeholder(api_key):
        raise ValueError("OpenRouter API key is not configured.")

    model = _env("OPENROUTER_MODEL", DEFAULT_MODEL)
    connect_timeout = float(_env("OPENROUTER_CONNECT_TIMEOUT", "5") or "5")
    read_timeout = float(_env("OPENROUTER_READ_TIMEOUT", "30") or "30")
    max_attempts = max(1, min(int(_env("OPENROUTER_MAX_RETRIES", "2") or "2") + 1, 4))

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,
    }
    headers = {
        "Authorization": "Bearer " + api_key,
        "Content-Type": "application/json",
    }

    session = get_http_session()
    last_error: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = session.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                json=payload,
                headers=headers,
                timeout=(connect_timeout, read_timeout),
            )
            if response.status_code in (429, 503) and attempt < max_attempts:
                time.sleep(0.4 * attempt)
                continue
            return response
        except requests.Timeout as exc:
            last_error = TimeoutError("OpenRouter request timed out.")
            if attempt < max_attempts:
                time.sleep(0.4 * attempt)
                continue
            raise last_error from exc
        except requests.RequestException as exc:
            last_error = ConnectionError("OpenRouter connection failed.")
            if attempt < max_attempts:
                time.sleep(0.4 * attempt)
                continue
            raise last_error from exc

    if last_error:
        raise last_error
    raise RuntimeError("OpenRouter request failed.")


def generate_answer(question: str, context: str) -> str:
    context_for_prompt = context if context else "No relevant context found in the indexed medical reference."
    messages = [
        {"role": "system", "content": system_prompt.format(context=context_for_prompt)},
        {"role": "user", "content": question},
    ]

    response = _openrouter_chat_completion(messages)
    if response.status_code in (401, 403):
        raise PermissionError("OpenRouter authentication failed.")
    if response.status_code == 429:
        raise RuntimeError("OpenRouter rate limit reached.")
    if response.status_code in (502, 503, 504):
        raise RuntimeError("OpenRouter service is temporarily unavailable.")
    if response.status_code >= 400:
        raise RuntimeError("OpenRouter returned an unexpected error.")

    try:
        payload = response.json()
        choices = payload.get("choices", [])
        content = choices[0].get("message", {}).get("content", "") if choices else ""
    except Exception as exc:
        raise ValueError("Malformed response from OpenRouter.") from exc

    answer = content.strip() if isinstance(content, str) else ""
    if not answer:
        raise ValueError("Malformed response from OpenRouter.")

    disclaimer = "\n\nMedical information only; this is not a diagnosis. Consult a qualified clinician."
    emergency_terms = {
        "chest pain",
        "stroke",
        "severe bleeding",
        "suicidal",
        "difficulty breathing",
        "heart attack",
        "unconscious",
    }
    lowered = question.lower()
    if any(term in lowered for term in emergency_terms):
        answer += "\n\nIf this may be an emergency, seek immediate local emergency care now."

    return (answer + disclaimer)[:6000]


def _extract_message() -> str:
    if request.is_json:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            raise ValueError("Invalid JSON payload.")
        msg = payload.get("msg", "")
    elif request.mimetype in {"application/x-www-form-urlencoded", "multipart/form-data", ""}:
        msg = request.form.get("msg", "")
    else:
        raise TypeError("Unsupported content type.")

    if not isinstance(msg, str):
        raise ValueError("Question must be text.")

    msg = msg.strip()
    max_input_chars = max(1, int(_env("MAX_INPUT_CHARS", "2000") or "2000"))
    if not msg:
        raise ValueError("Please enter a question.")
    if len(msg) > max_input_chars:
        raise OverflowError(f"Question exceeds {max_input_chars} characters.")
    return msg


@app.route("/")
def index() -> str:
    return render_template("chat.html")


@app.route("/health")
def health():
    return jsonify({"status": "healthy", "service": "medical-chatbot"}), 200


@app.route("/ready")
def ready():
    config = _required_config_status()
    if not all(config.values()):
        return jsonify({"error": "Service configuration is incomplete."}), 503

    try:
        client = get_pinecone_client()
        index_name = _env("PINECONE_INDEX_NAME", DEFAULT_INDEX_NAME)
        desc = client.describe_index(index_name)
        status = desc.get("status", {}) if isinstance(desc, dict) else getattr(desc, "status", {})
        ready_flag = status.get("ready", True) if isinstance(status, dict) else getattr(status, "ready", True)
        if not ready_flag:
            return jsonify({"error": "Pinecone index is not ready."}), 503
    except Exception:
        logger.exception("Readiness check failed")
        return jsonify({"error": "Service dependencies are not ready."}), 503

    return jsonify({"status": "ready", "service": "medical-chatbot"}), 200


@app.route("/get", methods=["POST"])
def chat():
    try:
        message = _extract_message()
    except TypeError:
        return jsonify({"error": "Unsupported content type."}), 415
    except ValueError:
        return jsonify({"error": "Invalid question input."}), 400
    except OverflowError:
        max_input_chars = max(1, int(_env("MAX_INPUT_CHARS", "2000") or "2000"))
        return jsonify({"error": f"Question exceeds {max_input_chars} characters."}), 413

    try:
        context = retrieve_context(message)
        answer = generate_answer(message, context)
        return jsonify({"answer": answer}), 200
    except PermissionError:
        logger.exception("Authentication error in request flow")
        return jsonify({"error": "Authentication with an upstream service failed."}), 502
    except TimeoutError:
        logger.exception("Timeout error in request flow")
        return jsonify({"error": "An upstream service timed out."}), 504
    except (ConnectionError, RuntimeError, ValueError):
        logger.exception("Upstream service error in request flow")
        return jsonify({"error": "An upstream service is currently unavailable."}), 502
    except Exception:
        logger.exception("Unexpected request failure")
        return jsonify({"error": "Unexpected server error."}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=False)
