import json
import unittest

import httpx

from app.agents.planner import PlannerError, build_planner_node, plan_research
from app.config import Settings
from app.llm import ChatCompletionsModel, ModelError
from app.state import create_initial_state, validate_state


VALID = json.dumps({"tasks": [{"question": "Costs?", "query": "cost comparison", "tool": "web_search"}]})


class SequenceModel:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = 0
        self.users = []

    def generate(self, **kwargs):
        self.calls += 1
        self.users.append(kwargs["user"])
        output = next(self.outputs)
        if isinstance(output, Exception):
            raise output
        return output


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(_env_file=None, planner_max_attempts=2, planner_max_tasks=5)

    def test_node_does_not_mutate_state(self):
        state = create_initial_state("Compare costs")
        update = build_planner_node(SequenceModel([VALID]), self.settings)(state)
        self.assertIsNone(state["research_plan"])
        self.assertEqual(update["current_task"].id, "task-1")
        validate_state({**state, **update})
        with self.assertRaises(PlannerError):
            build_planner_node(SequenceModel([VALID]), self.settings)({**state, **update})

    def test_invalid_output_can_be_repaired(self):
        for invalid in ("not JSON", '{"tasks": []}', '{"tasks": [{"question":" ","query":"x","tool":"web_search"}]}'):
            with self.subTest(invalid=invalid):
                model = SequenceModel([invalid, VALID])
                plan_research("question", model, self.settings)
                self.assertEqual(model.calls, 2)
                self.assertTrue(json.loads(model.users[-1])["validation_feedback"])

    def test_constraints_and_bounded_retries(self):
        local = VALID.replace("web_search", "retrieve_documents")
        for output in (local, json.dumps({"tasks": [json.loads(VALID)["tasks"][0]] * 6})):
            model = SequenceModel([output, output])
            with self.assertRaises(PlannerError):
                plan_research("question", model, self.settings)
            self.assertEqual(model.calls, 2)
        self.assertEqual(plan_research("question", SequenceModel([local]), self.settings,
                                      local_documents_available=True).tasks[0].tool, "retrieve_documents")

    def test_api_retry_policy(self):
        model = SequenceModel([ModelError("timeout", retryable=True), VALID])
        plan_research("question", model, self.settings)
        fatal = SequenceModel([ModelError("unauthorized")])
        with self.assertRaises(PlannerError):
            plan_research("question", fatal, self.settings)
        self.assertEqual(fatal.calls, 1)

    def test_transport_and_envelope_failures(self):
        settings = Settings(_env_file=None, api_key="fixture-secret", model_name="fixture-model",
                            api_base_url="https://api.openai.com/v1")
        def good(request):
            body = json.loads(request.content)
            self.assertTrue(body["response_format"]["json_schema"]["strict"])
            self.assertEqual(request.url.path, "/v1/chat/completions")
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": VALID}}]})
        model = ChatCompletionsModel(settings, transport=httpx.MockTransport(good))
        plan_research("question", model, self.settings)
        for status, payload, retryable in (
            (401, {}, False), (429, {}, True), (500, {}, True),
            (200, {"choices": []}, True),
            (200, {"choices": [{"finish_reason": "stop", "message": {"refusal": "No"}}]}, False),
            (200, {"choices": [{"finish_reason": "length", "message": {"content": VALID}}]}, True),
            (200, {"choices": [{"finish_reason": "stop", "message": {"content": ""}}]}, True),
        ):
            with self.subTest(status=status, payload=payload):
                transport = httpx.MockTransport(lambda req: httpx.Response(status, json=payload))
                with self.assertRaises(ModelError) as error:
                    ChatCompletionsModel(settings, transport=transport).generate(system="x", user="x", schema={})
                self.assertEqual(error.exception.retryable, retryable)

    def test_glm_json_mode_and_fenced_output(self):
        settings = Settings(_env_file=None, api_key="fixture-secret", model_name="glm-4.5-flash")
        def respond(request):
            body = json.loads(request.content)
            self.assertEqual(body["response_format"], {"type": "json_object"})
            self.assertEqual(body["thinking"], {"type": "disabled"})
            self.assertIn('"properties"', body["messages"][0]["content"])
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {
                "content": "```json\n" + VALID + "\n```",
            }}]})
        model = ChatCompletionsModel(settings, transport=httpx.MockTransport(respond))
        self.assertEqual(plan_research("question", model, settings).tasks[0].id, "task-1")

    def test_final_error_keeps_safe_reason(self):
        with self.assertRaisesRegex(PlannerError, "json_invalid"):
            plan_research("question", SequenceModel(["bad", "bad"]), self.settings)
        with self.assertRaisesRegex(PlannerError, "timed out"):
            plan_research("question", SequenceModel([
                ModelError("timed out", retryable=True), ModelError("timed out", retryable=True),
            ]), self.settings)


if __name__ == "__main__":
    unittest.main()
