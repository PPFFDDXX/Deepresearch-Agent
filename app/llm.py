"""Replaceable structured-output transport; no research decisions here."""

import json
from typing import Any, Protocol

import httpx

from app.config import Settings


class ModelError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class StructuredModel(Protocol):
    def generate(self, *, system: str, user: str, schema: dict[str, Any]) -> str:
        """Return JSON text or raise ModelError."""
        ...


class ChatCompletionsModel:
    """GLM uses JSON mode plus local validation; other models use strict schema."""

    def __init__(self, settings: Settings, *, transport: httpx.BaseTransport | None = None,
                 timeout_seconds: float | None = None):
        if settings.api_key is None or not settings.api_key.get_secret_value().strip():
            raise ModelError("Configure DR_API_KEY before using the live model")
        if settings.model_name == "your-model-name":
            raise ModelError("Configure DR_MODEL_NAME before using the live model")
        self.settings = settings
        self.transport = transport
        self.timeout_seconds = settings.llm_timeout_seconds if timeout_seconds is None else timeout_seconds
        if not 0 < self.timeout_seconds <= 300:
            raise ValueError("Model timeout must be between 0 and 300 seconds")

    def generate(self, *, system: str, user: str, schema: dict[str, Any]) -> str:
        settings = self.settings
        is_glm = settings.model_name.lower().startswith("glm-")
        if is_glm:
            system += "\nReturn one JSON object matching this JSON Schema:\n" + json.dumps(schema, ensure_ascii=False)
            response_format = {"type": "json_object"}
        else:
            response_format = {"type": "json_schema", "json_schema": {
                "name": "research_plan", "strict": True, "schema": schema,
            }}
        payload = {
            "model": settings.model_name,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "response_format": response_format,
        }
        if settings.model_name.lower() == "glm-4.5-flash":
            payload["thinking"] = {"type": "disabled"}
        try:
            with httpx.Client(timeout=httpx.Timeout(self.timeout_seconds, connect=min(10.0, self.timeout_seconds)),
                              transport=self.transport) as client:
                response = client.post(
                    settings.api_base_url.rstrip("/") + "/chat/completions",
                    headers={"Authorization": f"Bearer {settings.api_key.get_secret_value()}"},
                    json=payload,
                )
        except httpx.TimeoutException as exc:
            raise ModelError(
                f"Model request timed out ({type(exc).__name__}; read timeout={self.timeout_seconds:g}s)",
                retryable=True,
            ) from None
        except httpx.RequestError:
            raise ModelError("Model connection failed", retryable=True) from None
        if response.is_error:
            raise ModelError(
                f"Model API returned HTTP {response.status_code}",
                retryable=response.status_code in (408, 429) or response.status_code >= 500,
            )
        try:
            choice = response.json()["choices"][0]
            message = choice["message"]
            if message.get("refusal"):
                raise ModelError("Model declined the planning request")
            if choice.get("finish_reason") != "stop":
                reason = choice.get("finish_reason")
                label = reason if reason in ("length", "content_filter", "tool_calls", None) else "unexpected"
                raise ModelError(f"Model response was incomplete (finish_reason={label})", retryable=True)
            content = message["content"]
            if not isinstance(content, str) or not content.strip():
                raise ModelError("Model returned empty content", retryable=True)
            if is_glm:
                # Accept only an entire fenced JSON block, never extract arbitrary prose.
                lines = content.strip().splitlines()
                if len(lines) >= 3 and lines[0].lower() in ("```json", "```") and lines[-1] == "```":
                    content = "\n".join(lines[1:-1])
            return content
        except (ValueError, KeyError, IndexError, TypeError):
            raise ModelError("Model API returned an invalid response envelope", retryable=True) from None
