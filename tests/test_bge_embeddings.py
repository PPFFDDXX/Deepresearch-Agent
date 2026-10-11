"""Offline BGE-M3 API contract tests; never call the real service or main.py."""

import json
import unittest
from pathlib import Path

import httpx

from app.config import PROJECT_ROOT, Settings
from app.rag.embeddings import ApiEmbeddings, HashingEmbeddings, build_embeddings
from app.rag.models import RagError
from app.rag.pipeline import build_local_rag


class BgeEmbeddingTests(unittest.TestCase):
    def settings(self, **changes):
        return Settings(_env_file=None, **{
            "embedding_provider": "api", "embedding_model": "BAAI/bge-m3",
            "embedding_base_url": "https://api.siliconflow.cn/v1", "embedding_api_key": "fixture",
            **changes,
        })

    def response(self, texts, dimensions=1024):
        vectors = HashingEmbeddings(dimensions).embed(texts)
        return httpx.Response(200, json={"data": [
            {"index": index, "embedding": vectors[index]} for index in reversed(range(len(texts)))
        ]})

    def test_siliconflow_contract_and_batch_order(self):
        calls = []
        def handler(request):
            payload = json.loads(request.content)
            self.assertEqual(str(request.url), "https://api.siliconflow.cn/v1/embeddings")
            self.assertEqual(request.headers["Authorization"], "Bearer fixture")
            self.assertEqual(payload["model"], "BAAI/bge-m3")
            self.assertEqual(payload["encoding_format"], "float")
            self.assertNotIn("dimensions", payload)
            calls.append(payload["input"])
            return self.response(payload["input"])
        model = ApiEmbeddings(self.settings(embedding_batch_size=2), transport=httpx.MockTransport(handler))
        texts = ["现行知识库审计", "试点预算上限", "有答案检索命中率"]
        self.assertEqual(model.embed(texts), HashingEmbeddings(1024).embed(texts))
        self.assertEqual(calls, [texts[:2], texts[2:]])
        self.assertEqual(model.dimensions, 1024)

    def test_bge_wrong_dimensions_fail_before_indexing(self):
        for name in ("BAAI/bge-m3", "Pro/BAAI/bge-m3"):
            model = ApiEmbeddings(self.settings(embedding_model=name), transport=httpx.MockTransport(
                lambda request: self.response(["文档"], dimensions=768)))
            with self.subTest(name=name), self.assertRaisesRegex(RagError, "expected 1024, received 768"):
                model.embed(["文档"])

    def test_incompatible_bge_expected_dimension_is_rejected(self):
        with self.assertRaisesRegex(RagError, "1024"):
            ApiEmbeddings(self.settings(embedding_expected_dimensions=768))

    def test_generic_model_locks_dimensions_across_calls(self):
        responses = iter([self.response(["文档"], 128), self.response(["查询"], 256)])
        model = ApiEmbeddings(self.settings(embedding_model="fixture-model"), transport=httpx.MockTransport(lambda request: next(responses)))
        model.embed(["文档"])
        with self.assertRaisesRegex(RagError, "expected 128, received 256"):
            model.embed(["查询"])

    def test_generic_model_expected_dimension_is_enforced(self):
        model = ApiEmbeddings(self.settings(embedding_model="fixture-model", embedding_expected_dimensions=128),
            transport=httpx.MockTransport(lambda request: self.response(["查询"], 256)))
        with self.assertRaisesRegex(RagError, "expected 128, received 256"):
            model.embed(["查询"])

    def test_empty_input_validation_never_sends_requests(self):
        def forbidden(request):
            self.fail("Invalid input must not invoke HTTP")
        model = ApiEmbeddings(self.settings(), transport=httpx.MockTransport(forbidden))
        self.assertEqual(model.embed([]), [])
        for invalid in ("text", [" "], [None], ["valid", ""]):
            with self.subTest(invalid=invalid), self.assertRaises(RagError):
                model.embed(invalid)

    def test_missing_key_fails_without_leaking_details(self):
        with self.assertRaisesRegex(RagError, "DR_EMBEDDING_API_KEY"):
            ApiEmbeddings(self.settings(embedding_api_key=None))

    def test_api_selection_and_offline_override(self):
        self.assertIsInstance(build_embeddings(self.settings()), ApiEmbeddings)
        self.assertIsInstance(build_embeddings(self.settings(embedding_provider="hashing")), HashingEmbeddings)

    def test_timeout_retry_is_bounded(self):
        calls = []
        def handler(request):
            calls.append(request)
            raise httpx.ReadTimeout("fixture", request=request)
        model = ApiEmbeddings(self.settings(), transport=httpx.MockTransport(handler))
        with self.assertRaisesRegex(RagError, "attempts=2"):
            model.embed(["查询"])
        self.assertEqual(len(calls), 2)

    def test_mocked_bge_transport_connects_documents_faiss_and_queries(self):
        calls = []
        def handler(request):
            texts = json.loads(request.content)["input"]
            calls.append(texts)
            return self.response(texts)
        model = ApiEmbeddings(self.settings(), transport=httpx.MockTransport(handler))
        rag = build_local_rag([PROJECT_ROOT / "examples" / "doc"], self.settings(), embeddings=model)
        cases = [
            ("现行规范文档复核周期和审计日志保留时间", "02-knowledge-policy-current.md", ("30", "180")),
            ("三个月试点预算上限 人民币", "04-pilot-budget.txt", ("18,000",)),
            ("有答案问题的检索命中率 实测结果", "03-pilot-results.md", ("92.5%",)),
        ]
        for query, filename, terms in cases:
            with self.subTest(query=query):
                hits = rag.store.search(query, top_k=4, min_score=0.05)
                self.assertTrue(any(
                    Path(hit.chunk.locator).name == filename
                    and all(term in hit.chunk.text for term in terms)
                    for hit in hits
                ))
        self.assertEqual(len(rag.documents), 4)
        self.assertEqual(len(rag.store.chunks), 20)
        self.assertEqual(rag.store.index.d, 1024)
        self.assertEqual(len(calls), 5)


if __name__ == "__main__":
    unittest.main()
