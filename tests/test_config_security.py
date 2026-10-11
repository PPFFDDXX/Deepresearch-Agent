"""Configuration must work offline and never depend on embedded credentials."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import SecretStr

from app.config import Settings
from app.llm import ChatCompletionsModel, ModelError
from app.rag.embeddings import ApiEmbeddings
from app.rag.models import RagError
from app.tools.models import ToolError
from app.tools.web_search import TavilySearchBackend


class ConfigSecurityTests(unittest.TestCase):
    def test_clean_environment_has_no_default_credentials(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None)
        for name in ("api_key", "tavily_api_key", "embedding_api_key"):
            self.assertIsNone(getattr(settings, name))
        with self.assertRaisesRegex(ModelError, "DR_API_KEY"):
            ChatCompletionsModel(settings)
        with self.assertRaisesRegex(ToolError, "DR_TAVILY_API_KEY"):
            TavilySearchBackend(settings)
        with self.assertRaisesRegex(RagError, "DR_EMBEDDING_API_KEY"):
            ApiEmbeddings(settings)

    def test_blank_and_whitespace_keys_are_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None, api_key=" ", tavily_api_key=SecretStr("\n"), embedding_api_key="")
        self.assertIsNone(settings.api_key)
        self.assertIsNone(settings.tavily_api_key)
        self.assertIsNone(settings.embedding_api_key)

    def test_environment_precedence_and_secret_masking(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("DR_API_KEY=fixture-file\nDR_EMBEDDING_API_KEY=fixture-embedding\n", encoding="utf-8")
            with patch.dict(os.environ, {"DR_API_KEY": " fixture-env "}, clear=True):
                settings = Settings(_env_file=path)
            self.assertEqual(settings.api_key.get_secret_value(), "fixture-env")
            self.assertEqual(settings.embedding_api_key.get_secret_value(), "fixture-embedding")
            self.assertNotIn("fixture-env", repr(settings))
            self.assertNotIn("fixture-embedding", settings.model_dump_json())


if __name__ == "__main__":
    unittest.main()
