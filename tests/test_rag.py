"""Real demonstrator document retrieval, FAISS, HTTP failure and graph integration."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx
import numpy as np
from pydantic import ValidationError

from app.agents.planner import PlannerError
from app.config import PROJECT_ROOT, Settings
from app.graph import run_research
from app.rag.chunker import chunk_document
from app.rag.embeddings import ApiEmbeddings, HashingEmbeddings
from app.rag.loader import load_documents
from app.rag.models import RagError
from app.rag.pipeline import build_local_rag
from app.rag.vector_store import FaissVectorStore
from app.tools.models import ToolError
from examples.graph_demo import DemoGraphModel
from examples.rag_demo import run_demo
from examples.researcher_demo import demo_search
from main import build_live_search


DOCS = PROJECT_ROOT / "examples" / "doc"


class RagTests(unittest.TestCase):
    def settings(self, **changes):
        values = {"embedding_provider": "hashing", "embedding_model": "fixture", **changes}
        return Settings(_env_file=None, **values)

    def test_demo_document_ingestion_and_metadata(self):
        documents = load_documents([DOCS], self.settings())
        self.assertEqual(len(documents), 5)
        current = next(item for item in documents if "current" in item.locator)
        archived = next(item for item in documents if "archived" in item.locator)
        self.assertEqual(current.version, "2.0")
        self.assertEqual(current.effective_date, "2026-09-01")
        self.assertFalse(current.archived)
        self.assertTrue(archived.archived)
        rag = build_local_rag([DOCS], self.settings())
        self.assertEqual(len(rag.documents), 4)
        self.assertEqual(rag.excluded_archived, 1)
        self.assertNotIn("archived.md", rag.describe_documents())

    def test_known_current_policy_and_budget_retrieve_correct_sections(self):
        rag = build_local_rag([DOCS], self.settings())
        policy = rag.retrieve_documents("现行知识库 审计日志保留 复核周期")
        self.assertIn("180", policy[0].text)
        self.assertIn("30", policy[0].text)
        self.assertIn("02-knowledge-policy-current.md", policy[0].locator)
        budget = rag.retrieve_documents("试点预算上限 人民币 18000")
        self.assertIn("18,000", budget[0].text)
        self.assertIn("04-pilot-budget.txt", budget[0].locator)
        self.assertTrue(all(item.origin == "local_chunk" and item.source_type == "local" for item in policy + budget))
        self.assertIn("version 2.0", policy[0].location)

    def test_archived_document_requires_explicit_opt_in(self):
        default = build_local_rag([DOCS], self.settings())
        historical = build_local_rag([DOCS], self.settings(), include_archived=True)
        self.assertFalse(any("archived.md" in chunk.locator for chunk in default.store.chunks))
        self.assertTrue(any("archived.md" in chunk.locator for chunk in historical.store.chunks))
        hits = historical.retrieve_documents("历史留存规定 每60天复核 审计日志90天")
        self.assertTrue(any("05-knowledge-policy-archived.md" in item.locator and "90" in item.text for item in hits))

    def test_chunk_windows_are_grounded_bounded_and_ids_are_stable(self):
        documents = load_documents([DOCS], self.settings())
        for document in documents:
            chunks = chunk_document(document, 100, 20)
            self.assertEqual(chunks, chunk_document(document, 100, 20))
            self.assertEqual(len(chunks), len({item.id for item in chunks}))
            for chunk in chunks:
                self.assertLessEqual(len(chunk.text), 100)
                self.assertTrue(any(chunk.text in page.text for page in document.pages))
                self.assertIn("section", chunk.location)
        with self.assertRaises(ValueError):
            chunk_document(documents[0], 100, 100)
        with self.assertRaises(ValidationError):
            self.settings(rag_chunk_size=100, rag_chunk_overlap=100)

    def test_empty_invalid_unreadable_and_oversized_inputs(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaises(RagError):
                load_documents([root], self.settings())
            for filename, content in (("empty.txt", b""), ("bad.txt", b"\xff\xfe\xff"), ("broken.pdf", b"broken")):
                path = root / filename
                path.write_bytes(content)
                with self.subTest(filename=filename), self.assertRaises(RagError):
                    load_documents([path], self.settings())
            huge = root / "huge.txt"
            huge.write_text("a" * (1024 * 1024 + 1), encoding="utf-8")
            with self.assertRaises(RagError):
                load_documents([huge], self.settings(rag_max_file_mb=1))
            with self.assertRaises(RagError):
                load_documents([root / "missing.md"], self.settings())
        with self.assertRaises(RagError):
            load_documents([DOCS], self.settings(rag_max_files=1))

    def test_file_deduplication_and_archive_only_input(self):
        file = DOCS / "01-company-overview.md"
        self.assertEqual(len(load_documents([file, file], self.settings())), 1)
        with self.assertRaises(RagError):
            build_local_rag([DOCS / "05-knowledge-policy-archived.md"], self.settings())

    def test_pdf_text_and_page_locations_and_scanned_pdf(self):
        from pypdf import PdfWriter
        from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
        with TemporaryDirectory() as folder:
            path = Path(folder) / "paper.pdf"
            writer = PdfWriter()
            for text in ("First page evidence", "Second page evidence"):
                page = writer.add_blank_page(width=300, height=300)
                font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                    NameObject("/Subtype"): NameObject("/Type1"), NameObject("/BaseFont"): NameObject("/Helvetica")})
                page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
                stream = DecodedStreamObject()
                stream.set_data(f"BT /F1 12 Tf 10 100 Td ({text}) Tj ET".encode())
                page[NameObject("/Contents")] = writer._add_object(stream)
            writer.write(path)
            documents = load_documents([path], self.settings())
            chunks = chunk_document(documents[0], 100, 20)
            self.assertEqual([page.page_number for page in documents[0].pages], [1, 2])
            self.assertIn("page 2", chunks[1].location)
            blank = PdfWriter()
            blank.add_blank_page(width=100, height=100)
            blank.write(path)
            with self.assertRaises(RagError):
                load_documents([path], self.settings())

    def test_hashing_is_deterministic_and_cosine_threshold_returns_empty(self):
        embeddings = HashingEmbeddings(256)
        vectors = embeddings.embed(["知识库审计日志", "知识库审计日志"])
        self.assertEqual(vectors[0], vectors[1])
        self.assertAlmostEqual(float(np.linalg.norm(vectors[0])), 1, places=5)
        rag = build_local_rag([DOCS], self.settings(rag_min_score=1))
        self.assertEqual(rag.retrieve_documents("completely unrelated vocabulary"), [])
        self.assertEqual(rag.retrieve_documents("!!!"), [])
        with self.assertRaises(ToolError) as raised:
            rag.retrieve_documents(" ")
        self.assertEqual(raised.exception.code, "invalid_input")

    def test_api_embeddings_batch_order_and_retry(self):
        calls = []
        def handler(request):
            inputs = json.loads(request.content)["input"]
            calls.append(inputs)
            if len(calls) == 1:
                return httpx.Response(429)
            return httpx.Response(200, json={"data": [
                {"index": index, "embedding": [float(index + 1), 1.0]} for index in reversed(range(len(inputs)))
            ]})
        model = ApiEmbeddings(self.settings(embedding_api_key="fixture", embedding_batch_size=2), transport=httpx.MockTransport(handler))
        self.assertEqual(model.embed(["one", "two", "three"]), [[1, 1], [2, 1], [1, 1]])
        self.assertEqual(len(calls), 3)

    def test_embedding_failures_and_index_dimensions(self):
        with self.assertRaises(RagError):
            ApiEmbeddings(self.settings(embedding_api_key=None))
        for body in ('{"data":[]}', '{"data":[{"index":0,"embedding":[0,0]}]}',
                     '{"data":[{"index":0,"embedding":[NaN,1]}]}'):
            model = ApiEmbeddings(self.settings(embedding_api_key="fixture"), transport=httpx.MockTransport(
                lambda request: httpx.Response(200, text=body)))
            with self.assertRaises(RagError):
                model.embed(["q"])
        calls = []
        def unauthorized(request):
            calls.append(request)
            return httpx.Response(401)
        model = ApiEmbeddings(self.settings(embedding_api_key="fixture"), transport=httpx.MockTransport(unauthorized))
        with self.assertRaises(RagError):
            model.embed(["q"])
        self.assertEqual(len(calls), 1)
        document = load_documents([DOCS / "01-company-overview.md"], self.settings())[0]
        class BadModel:
            def embed(self, texts):
                return [[1.0, 2.0]] * len(texts) if len(texts) > 1 else [[1.0]]
        store = FaissVectorStore(chunk_document(document, 100, 20), BadModel())
        with self.assertRaises(RagError):
            store.search("q", top_k=1, min_score=0)

    def test_local_demo_runs_graph_and_preserves_real_file_citations(self):
        rag = build_local_rag([DOCS], self.settings())
        result = run_demo(self.settings(), rag)
        self.assertEqual(result["report_status"], "generated")
        self.assertTrue(all(task["tool"] == "retrieve_documents" for task in result["research_plan"]["tasks"]))
        self.assertTrue(all(item["origin"] == "local_chunk" for item in result["evidence"]))
        self.assertTrue(all(Path(source["locator"]).is_file() for source in result["report_citations"]))
        self.assertIn("180", result["final_report"])

    def test_mixed_graph_chooses_local_and_web_tools(self):
        rag = build_local_rag([DOCS], self.settings())
        class MixedModel(DemoGraphModel):
            def generate(self, **kwargs):
                if "tasks" in kwargs["schema"]["properties"]:
                    payload = json.loads(kwargs["user"])
                    self_inventory = payload["local_document_inventory"]
                    if "02-knowledge-policy-current.md" not in self_inventory:
                        raise AssertionError("Planner did not receive the document inventory")
                    return json.dumps({"tasks": [
                        {"question": "Internal policy", "query": "审计日志留存", "tool": "retrieve_documents"},
                        {"question": "External fixture", "query": "fixture", "tool": "web_search"},
                    ]})
                return super().generate(**kwargs)
        state = run_research("Mixed question", MixedModel(), self.settings(), web_search=demo_search,
                             retrieve_documents=rag.retrieve_documents, local_document_context=rag.describe_documents())
        self.assertEqual({item.origin for item in state["evidence"]}, {"local_chunk", "search_snippet"})

    def test_local_only_rejects_planner_web_tasks_and_lazy_web_needs_no_key(self):
        rag = build_local_rag([DOCS], self.settings())
        tool = build_live_search(self.settings(), enabled=False)
        with self.assertRaises(ToolError):
            tool("query")
        with self.assertRaises(PlannerError):
            run_research("Local question", DemoGraphModel(), self.settings(), web_search=tool,
                         retrieve_documents=rag.retrieve_documents, web_search_available=False)


if __name__ == "__main__":
    unittest.main()
