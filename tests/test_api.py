"""Offline HTTP/job tests; never send requests to model providers."""
import time
import unittest
from pathlib import Path
from threading import Event
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.api import create_app
from app.config import Settings
from app.jobs import JobCapacityError, JobManager
from app.service import execute_research


class ApiTests(unittest.TestCase):
    def settings(self):
        return Settings(_env_file=None, api_key="fixture", embedding_provider="hashing", trace_file=None)

    def wait_job(self, client, job_id):
        for _ in range(200):
            result = client.get(f"/research/{job_id}").json()
            if result["status"] in ("completed", "failed"):
                return result
            time.sleep(0.01)
        self.fail("Job did not finish")

    def test_health_openapi_validation_and_missing_job(self):
        with TestClient(create_app(self.settings(), runner=lambda *a, **kw: {})) as client:
            self.assertEqual(client.get("/health").json()["phase"], 11)
            self.assertIn("/research/files", client.get("/openapi.json").json()["paths"])
            response = client.post("/research", json={"query": "SECRET", "local_only": True})
            self.assertEqual(response.status_code, 422)
            self.assertNotIn("SECRET", response.text)
            self.assertEqual(client.get("/research/missing").status_code, 404)

    def test_queued_running_report_and_isolated_requests(self):
        started, release = Event(), Event()
        def runner(query, settings, **kwargs):
            started.set()
            release.wait(2)
            return {"final_report": query, "research_status": "sufficient", "report_status": "generated"}
        with TestClient(create_app(self.settings(), runner=runner)) as client:
            response = client.post("/research", json={"query": "first"})
            self.assertEqual(response.status_code, 202)
            job_id = response.json()["job_id"]
            self.assertTrue(started.wait(1))
            self.assertEqual(client.get(f"/research/{job_id}/report").status_code, 409)
            release.set()
            job = self.wait_job(client, job_id)
            self.assertEqual(job["status"], "completed")
            self.assertIsNotNone(job["run_id"])
            self.assertEqual(client.get(f"/research/{job_id}/report").text, "first")
            second = client.post("/research", json={"query": "second"}).json()["job_id"]
            self.assertEqual(self.wait_job(client, second)["result"]["final_report"], "second")
            self.assertEqual(client.get(f"/research/{job_id}/report").text, "first")

    def test_uploaded_files_available_until_completion_and_removed_afterwards(self):
        seen = []
        def runner(query, settings, **kwargs):
            seen.extend(kwargs["documents"])
            self.assertTrue(kwargs["local_only"])
            self.assertEqual(seen[0].read_text(), "fixture")
            return {"final_report": "report", "report_status": "generated"}
        with TestClient(create_app(self.settings(), runner=runner)) as client:
            response = client.post("/research/files", data={"query": "question"}, files=[("files", ("source.md", b"fixture"))])
            self.assertEqual(response.status_code, 202)
            self.assertEqual(self.wait_job(client, response.json()["job_id"])["status"], "completed")
        self.assertFalse(seen[0].exists())

    def test_bad_files_and_empty_query(self):
        with TestClient(create_app(self.settings(), runner=lambda *a, **kw: {})) as client:
            for name, content, expected in [("../secret.txt", b"x", 422), ("a.exe", b"x", 422), ("a.md", b"", 413)]:
                response = client.post("/research/files", data={"query": "question"}, files=[("files", (name, content))])
                self.assertEqual(response.status_code, expected)
            response = client.post("/research/files", data={"query": "   "}, files=[("files", ("a.md", b"x"))])
            self.assertEqual(response.status_code, 422)

    def test_safe_failure_and_business_failure(self):
        def broken(*a, **kw):
            raise RuntimeError("SECRET")
        with TestClient(create_app(self.settings(), runner=broken)) as client:
            job_id = client.post("/research", json={"query": "question"}).json()["job_id"]
            result = self.wait_job(client, job_id)
            self.assertEqual(result["error_type"], "RuntimeError")
            self.assertNotIn("SECRET", str(result))
            self.assertEqual(client.get(f"/research/{job_id}/report").status_code, 409)
        with TestClient(create_app(self.settings(), runner=lambda *a, **kw: {"report_status": "writer_error", "final_report": "unavailable"})) as client:
            job_id = client.post("/research", json={"query": "question"}).json()["job_id"]
            self.assertEqual(self.wait_job(client, job_id)["status"], "failed")

    def test_capacity_and_snapshot(self):
        release = Event()
        manager = JobManager(self.settings(), lambda *a, **kw: release.wait(2) or {}, workers=1, max_pending=1)
        try:
            job = manager.submit("first")
            with self.assertRaises(JobCapacityError):
                manager.submit("second")
            snapshot = manager.get(job.job_id)
            snapshot.status = "failed"
            self.assertNotEqual(manager.get(job.job_id).status, "failed")
        finally:
            release.set()
            manager.close()

    def test_service_reuses_graph_and_rag_dependencies(self):
        from app.state import create_initial_state
        with patch("app.service.run_research", return_value=create_initial_state("question")) as graph:
            result = execute_research("question", self.settings())
            self.assertFalse(result["evaluation"]["semantic_support_verified"])
            self.assertTrue(graph.call_args.kwargs["web_search_available"])
        with self.assertRaises(ValueError):
            execute_research("question", self.settings(), local_only=True)


if __name__ == "__main__":
    unittest.main()
