"""Offline showcase must exercise the real graph without provider requests."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from app.config import PROJECT_ROOT
from examples.showcase import export_showcase, run_case


class ShowcaseTests(unittest.TestCase):
    def test_external_environment_cannot_change_offline_budgets(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            "DR_MAX_RETRIES": "0", "DR_EMBEDDING_PROVIDER": "api",
            "DR_TRACE_FILE": str(Path(directory) / "unexpected.jsonl"),
        }):
            state = run_case("partial")
            self.assertEqual(state["retry_count"], 2)
            self.assertFalse((Path(directory) / "unexpected.jsonl").exists())

    def test_exports_success_partial_and_refusal_without_http(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            httpx.Client, "post", side_effect=AssertionError("Offline showcase must not send HTTP")
        ):
            output = Path(directory)
            manifest = export_showcase(output)
            self.assertEqual([item["report_status"] for item in manifest["cases"]], ["generated", "partial", "no_evidence"])
            for item in manifest["cases"]:
                state = json.loads((output / item["audit"]).read_text(encoding="utf-8"))
                report = (output / item["report"]).read_text(encoding="utf-8")
                self.assertTrue(state["evaluation"]["state_valid"])
                self.assertTrue(state["research_context"]["requirement_checks"])
                self.assertIn("OFFLINE FIXTURE", report)
                self.assertNotIn(str(PROJECT_ROOT), json.dumps(state, ensure_ascii=False))
                self.assertNotIn("D:\\\\PFDX", report)
                self.assertEqual(report.split("\n\n", 1)[1], state["final_report"])
            success = (output / "success.md").read_text(encoding="utf-8")
            self.assertIn("18,000", success)
            self.assertIn("6,000", success)
            partial = json.loads((output / "partial.json").read_text(encoding="utf-8"))
            self.assertEqual(partial["retry_count"], 2)
            self.assertTrue(any(task["refinement_origin"] == "automatic" for task in partial["research_plan"]["tasks"]))
            refusal = json.loads((output / "refusal.json").read_text(encoding="utf-8"))
            self.assertEqual(refusal["report_citations"], [])
            self.assertIsNone(refusal["report_draft"])


if __name__ == "__main__":
    unittest.main()
