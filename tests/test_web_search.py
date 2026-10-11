import json
import unittest
import httpx
from app.config import Settings
from app.tools.models import ToolError, WebSearchHit
from app.tools.web_search import TavilySearchBackend, build_web_search

HIT = {"title": "Fixture", "url": "https://example.org/page", "content": "Fixture snippet"}


class SearchTests(unittest.TestCase):
    def tool(self, handler, **kwargs):
        settings = Settings(_env_file=None, tavily_api_key="fixture-key", search_max_attempts=2, **kwargs)
        return build_web_search(TavilySearchBackend(settings, transport=httpx.MockTransport(handler)), settings)

    def test_request_and_deduplication(self):
        def respond(request):
            body = json.loads(request.content)
            self.assertEqual(body["query"], "question")
            self.assertFalse(body["include_answer"])
            self.assertEqual(str(request.url), "https://api.tavily.com/search")
            self.assertEqual(request.headers["Authorization"], "Bearer fixture-key")
            return httpx.Response(200, json={"results": [HIT, {**HIT, "url": HIT["url"] + "#section"},
                                                        {**HIT, "url": HIT["url"] + "?version=2"}]})
        result = self.tool(respond)(" question ")
        self.assertEqual(len(result.results), 2)
        self.assertEqual(result.attempts, 1)

    def test_empty_and_invalid_input(self):
        result = self.tool(lambda req: httpx.Response(200, json={"results": []}))("question")
        self.assertEqual(result.results, [])
        tool = self.tool(lambda req: self.fail("Invalid input sent a request"))
        for query in (" ", None, 123):
            with self.assertRaises(ToolError) as error:
                tool(query)
            self.assertEqual(error.exception.code, "invalid_input")
        with self.assertRaises(ToolError):
            # Exercise the adapter's defensive check independently of Settings constraints.
            TavilySearchBackend(Settings(_env_file=None, tavily_api_key="fixture").model_copy(
                update={"tavily_api_key": None}))

    def test_retry_success_and_http_failure(self):
        calls = []
        def respond(request):
            calls.append(request)
            return httpx.Response(429 if len(calls) == 1 else 200,
                                  json={} if len(calls) == 1 else {"results": [HIT]})
        self.assertEqual(self.tool(respond)("question").attempts, 2)
        for status, attempts in ((401, 1), (400, 1), (500, 2), (429, 2)):
            with self.subTest(status=status):
                with self.assertRaises(ToolError) as error:
                    self.tool(lambda req: httpx.Response(status, text="private body"))("question")
                self.assertEqual(error.exception.attempts, attempts)
                self.assertNotIn("private body", str(error.exception))

    def test_timeout_and_network(self):
        for exception, code in ((httpx.ReadTimeout, "timeout"), (httpx.ConnectError, "network")):
            def fail(request):
                raise exception("private details", request=request)
            with self.assertRaises(ToolError) as error:
                self.tool(fail)("question")
            self.assertEqual(error.exception.code, code)
            self.assertEqual(error.exception.attempts, 2)

    def test_malformed_response(self):
        for payload in ({}, {"results": None}, {"results": [{**HIT, "url": "file:///tmp/a"}]},
                        {"results": [{**HIT, "content": " "}]}, {"results": [None]}):
            with self.subTest(payload=payload):
                with self.assertRaises(ToolError) as error:
                    self.tool(lambda req: httpx.Response(200, json=payload))("question")
                self.assertEqual(error.exception.code, "invalid_response")
        with self.assertRaises(ToolError):
            self.tool(lambda req: httpx.Response(200, text="not JSON"))("question")

    def test_replaceable_backend_and_limit(self):
        class OtherBackend:
            name = "other"
            def search(self, query, *, top_k, timeout):
                return [WebSearchHit(title="Fixture", url=f"https://example.org/{i}", snippet="fixture")
                        for i in range(4)]
        settings = Settings(_env_file=None, search_top_k=2)
        result = build_web_search(OtherBackend(), settings)("question")
        self.assertEqual(result.provider, "other")
        self.assertEqual(len(result.results), 2)


if __name__ == "__main__":
    unittest.main()
