"""Decoupled web_search(query) interface with a Tavily backend."""

from collections.abc import Callable
from typing import Protocol

import httpx
from pydantic import ValidationError

from app.config import Settings
from app.tools.models import ToolError, WebSearchHit, WebSearchResponse


class SearchBackend(Protocol):
    name: str

    def search(self, query: str, *, top_k: int, timeout: float) -> list[WebSearchHit]:
        """Perform one attempt; return validated hits or raise ToolError."""
        ...


class TavilySearchBackend:
    name = "tavily"

    def __init__(self, settings: Settings, *, transport: httpx.BaseTransport | None = None):
        if settings.tavily_api_key is None or not settings.tavily_api_key.get_secret_value().strip():
            raise ToolError("configuration", "Configure DR_TAVILY_API_KEY to enable live search")
        self.api_key = settings.tavily_api_key
        self.transport = transport

    def search(self, query: str, *, top_k: int, timeout: float) -> list[WebSearchHit]:
        try:
            with httpx.Client(timeout=timeout, transport=self.transport) as client:
                response = client.post(
                    "https://api.tavily.com/search",
                    headers={"Authorization": f"Bearer {self.api_key.get_secret_value()}"},
                    json={"query": query, "max_results": top_k, "search_depth": "basic",
                          "include_answer": False, "include_raw_content": False},
                )
        except httpx.TimeoutException:
            raise ToolError("timeout", "Search request timed out", retryable=True) from None
        except httpx.RequestError:
            raise ToolError("network", "Search connection failed", retryable=True) from None
        if response.is_error:
            raise ToolError("http", f"Search API returned HTTP {response.status_code}",
                            retryable=response.status_code in (408, 429) or response.status_code >= 500)
        try:
            payload = response.json()
            if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
                raise ValueError("Missing results list")
            hits = []
            for item in payload["results"]:
                hits.append(WebSearchHit(title=item["title"], url=item["url"], snippet=item["content"]))
            return hits
        except (ValueError, KeyError, TypeError, ValidationError):
            raise ToolError("invalid_response", "Search API returned malformed results", retryable=True) from None


def build_web_search(backend: SearchBackend, settings: Settings) -> Callable[[str], WebSearchResponse]:
    """Bind dependencies once; consumers call the returned web_search(query)."""

    def web_search(query: str) -> WebSearchResponse:
        if not isinstance(query, str) or not query.strip():
            raise ToolError("invalid_input", "Search query must be a nonempty string")
        cleaned = query.strip()
        for attempt in range(1, settings.search_max_attempts + 1):
            try:
                hits = backend.search(cleaned, top_k=settings.search_top_k,
                                      timeout=settings.tool_timeout_seconds)
                unique: dict[str, WebSearchHit] = {}
                for hit in hits:
                    # Drop fragments only: query parameters may identify different pages.
                    key = str(hit.url).split("#", 1)[0]
                    unique.setdefault(key, hit)
                return WebSearchResponse(query=cleaned, provider=backend.name,
                                         results=list(unique.values())[:settings.search_top_k], attempts=attempt)
            except ToolError as exc:
                if not exc.retryable or attempt == settings.search_max_attempts:
                    raise ToolError(exc.code, f"{exc} (attempts={attempt})",
                                    retryable=exc.retryable, attempts=attempt) from None
        raise AssertionError("Unreachable: search_max_attempts must be positive")

    return web_search
