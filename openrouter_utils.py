import json
import time
from urllib import error, request


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
APP_TITLE = "Chatbot Evaluator"
APP_REFERER = "http://localhost"
DEFAULT_TIMEOUT_SECONDS = 60
MAX_TIMEOUT_RETRIES = 2


def openrouter_chat_completion(
    api_key: str,
    model: str,
    messages: list[dict],
    temperature: float | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    if not api_key:
        raise ValueError("OpenRouter API key is missing.")

    payload_dict = {"model": model, "messages": messages}
    if temperature is not None:
        payload_dict["temperature"] = temperature

    payload = json.dumps(payload_dict).encode("utf-8")
    req = request.Request(
        OPENROUTER_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": APP_REFERER,
            "X-Title": APP_TITLE,
        },
        method="POST",
    )

    last_exception = None
    for attempt in range(MAX_TIMEOUT_RETRIES + 1):
        try:
            with request.urlopen(req, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            break
        except error.HTTPError as exc:
            raise RuntimeError(
                f"OpenRouter request failed: {exc.code} {exc.reason}"
            ) from exc
        except error.URLError as exc:
            last_exception = exc
            if "timed out" in str(exc.reason).lower() and attempt < MAX_TIMEOUT_RETRIES:
                continue
            raise RuntimeError(f"OpenRouter request failed: {exc.reason}") from exc
        except TimeoutError as exc:
            last_exception = exc
            if attempt < MAX_TIMEOUT_RETRIES:
                continue
            raise RuntimeError(f"OpenRouter request failed: {exc}") from exc
        except Exception as exc:
            last_exception = exc
            if "timed out" in str(exc).lower() and attempt < MAX_TIMEOUT_RETRIES:
                continue
            raise RuntimeError(f"OpenRouter request failed: {exc}") from exc
    else:
        raise RuntimeError(f"OpenRouter request failed: {last_exception}")

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("OpenRouter response format was unexpected.") from exc


def timed_openrouter_chat_completion(
    api_key: str,
    model: str,
    messages: list[dict],
    temperature: float | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict:
    started_at = time.perf_counter()
    try:
        content = openrouter_chat_completion(
            api_key=api_key,
            model=model,
            messages=messages,
            temperature=temperature,
            timeout=timeout,
        )
        return {
            "model": model,
            "content": content,
            "error": "",
            "duration_seconds": time.perf_counter() - started_at,
        }
    except Exception as exc:
        return {
            "model": model,
            "content": "",
            "error": str(exc),
            "duration_seconds": time.perf_counter() - started_at,
        }
