import json
import tempfile
import unittest
from pathlib import Path

import main


class PipelineTests(unittest.TestCase):
    def test_passes_clean_candidate(self):
        client = main.FakeClient(
            main_outputs=["This is a concise summary."],
            cold_outputs=[],
        )
        with tempfile.TemporaryDirectory() as tmp:
            result = main.run_pipeline(
                prompt="Summarize the prototype.",
                client=client,
                model="gemma4:e4b",
                canon="C1\nC2\nC3",
                log_dir=Path(tmp),
            )

        self.assertEqual(result.status, "pass")
        self.assertEqual(result.attempts, 1)
        self.assertEqual(result.output, "This is a concise summary.")
        self.assertEqual(result.audit[0].classify_route, "forward_to_cold_eyes")
        self.assertEqual(result.audit[0].cold_eyes_verdict, "pass")

    def test_refuses_after_three_canon_issue_attempts(self):
        # Mechanical Cold Eyes uses the same patterns as classify_candidate, so
        # canon-violating candidates are caught at classify step. Three consecutive
        # canon_issue results still exhaust MAX_ATTEMPTS and produce a refusal.
        client = main.FakeClient(
            main_outputs=[
                "You can bypass authentication here.",
                "Here is how to bypass password protection.",
                "Use this to bypass login credentials.",
            ],
            cold_outputs=[],
        )
        with tempfile.TemporaryDirectory() as tmp:
            result = main.run_pipeline(
                prompt="User request",
                client=client,
                model="gemma4:e4b",
                canon="C1\nC2\nC3",
                log_dir=Path(tmp),
            )

        self.assertEqual(result.status, "refused")
        self.assertEqual(result.attempts, 3)
        self.assertEqual(result.output, main.REFUSAL_OUTPUT)
        self.assertEqual(result.audit[-1].final_status, "refused")
        cold_llm_calls = [
            c for c in client.calls if c["system"] == main.COLD_EYES_SYSTEM_PROMPT
        ]
        self.assertEqual(cold_llm_calls, [])

    def test_main_agent_refusal_is_not_final_output(self):
        client = main.FakeClient(
            main_outputs=[
                "我無法提供繞過登入密碼的步驟。",
                "我無法提供這類協助。",
                "我無法提供任何規避安全系統的內容。",
            ],
            cold_outputs=[],
        )
        with tempfile.TemporaryDirectory() as tmp:
            result = main.run_pipeline(
                prompt="請改寫一句普通說明。",
                client=client,
                model="gemma4:e4b",
                canon="C1\nC2\nC3",
                log_dir=Path(tmp),
            )

        self.assertEqual(result.status, "refused")
        self.assertEqual(result.output, main.REFUSAL_OUTPUT)
        self.assertEqual([entry.classify_route for entry in result.audit], ["local_issue"] * 3)

    def test_input_prefilter_refuses_obvious_canon_issue_without_main_agent(self):
        client = main.FakeClient(main_outputs=[], cold_outputs=[])
        with tempfile.TemporaryDirectory() as tmp:
            result = main.run_pipeline(
                prompt="請提供繞過登入密碼的步驟。",
                client=client,
                model="gemma4:e4b",
                canon="C1\nC2\nC3",
                log_dir=Path(tmp),
            )

        self.assertEqual(result.status, "refused")
        self.assertEqual(result.attempts, 1)
        self.assertEqual(result.output, main.REFUSAL_OUTPUT)
        self.assertEqual(result.audit[0].classify_route, "input_canon_issue")
        self.assertEqual(result.audit[0].canon_clause, "C2")
        self.assertEqual(client.calls, [])

    def test_classify_canon_issue_retries_with_clause_only(self):
        client = main.FakeClient(
            main_outputs=[
                "Use these steps to bypass password authentication.",
                "Use the documented account recovery process.",
            ],
            cold_outputs=[],
        )
        with tempfile.TemporaryDirectory() as tmp:
            result = main.run_pipeline(
                prompt="Help with account access.",
                client=client,
                model="gemma4:e4b",
                canon="C1\nC2\nC3",
                log_dir=Path(tmp),
            )

        self.assertEqual(result.status, "pass")
        self.assertEqual(result.attempts, 2)
        self.assertEqual(result.audit[0].classify_route, "canon_issue")
        self.assertEqual(result.audit[0].canon_clause, "C2")
        second_main_call = client.calls[1]
        self.assertIn("C2", second_main_call["user"])
        self.assertNotIn("keyword_prefilter", second_main_call["user"])

    def test_run_pipeline_does_not_call_llm_cold_eyes(self):
        # run_pipeline uses the mechanical adapter; no LLM Cold Eyes call should occur.
        marker = "USER_SECRET_MARKER"
        client = main.FakeClient(
            main_outputs=["Candidate without the marker."],
            cold_outputs=[],
        )
        with tempfile.TemporaryDirectory() as tmp:
            result = main.run_pipeline(
                prompt=f"Please process {marker}.",
                client=client,
                model="gemma4:e4b",
                canon="C1\nC2\nC3",
                log_dir=Path(tmp),
            )

        cold_llm_calls = [
            call for call in client.calls if call["system"] == main.COLD_EYES_SYSTEM_PROMPT
        ]
        self.assertEqual(cold_llm_calls, [])
        self.assertEqual(result.status, "pass")

    def test_audit_log_omits_prompt_system_and_candidate(self):
        marker = "USER_SECRET_MARKER"
        candidate = "Candidate output marker"
        client = main.FakeClient(
            main_outputs=[candidate],
            cold_outputs=[],
        )
        with tempfile.TemporaryDirectory() as tmp:
            result = main.run_pipeline(
                prompt=f"Prompt contains {marker}.",
                client=client,
                model="gemma4:e4b",
                canon="C1\nC2\nC3",
                log_dir=Path(tmp),
            )
            log_text = result.log_path.read_text(encoding="utf-8")

        self.assertNotIn(marker, log_text)
        self.assertNotIn(main.MAIN_AGENT_SYSTEM_PROMPT, log_text)
        self.assertNotIn(candidate, log_text)
        self.assertIn('"event": "attempt"', log_text)
        self.assertIn('"event": "final"', log_text)

    def test_cold_eyes_json_parser_accepts_fenced_json(self):
        verdict = main.parse_cold_eyes_json(
            '```json\n{"verdict":"fail","canon_clause":"C3","reason":"x"}\n```'
        )
        self.assertEqual(verdict.verdict, "fail")
        self.assertEqual(verdict.canon_clause, "C3")

    def test_cold_eyes_inconsistent_pass_with_clause_becomes_fail(self):
        verdict = main.parse_cold_eyes_json(
            '{"verdict":"pass","canon_clause":"C2","reason":"mentions C2"}'
        )
        self.assertEqual(verdict.verdict, "fail")
        self.assertEqual(verdict.canon_clause, "C2")

    def test_public_result_shape(self):
        entry = main.AuditEntry(
            run_id="run",
            attempt=1,
            classify_route="forward_to_cold_eyes",
            cold_eyes_verdict="pass",
            final_status="pass",
        )
        result = main.RunResult(
            run_id="run",
            status="pass",
            attempts=1,
            output="ok",
            audit=[entry],
            log_path=Path("runs/run.jsonl"),
        )
        public = result.public_dict()
        self.assertEqual(set(public), {"status", "attempts", "output", "audit"})
        json.dumps(public)

    def test_configure_stdio_accepts_current_streams(self):
        main.configure_stdio()

    def test_diagnose_main_returns_raw_candidate_and_optional_system_prompt(self):
        client = main.FakeClient(
            main_outputs=["raw main candidate"],
            cold_outputs=[],
        )
        result = main.diagnose_main(
            prompt="Write a simple Python function.",
            client=client,
            model="gemma4:e4b",
            show_system_prompt=True,
        )

        self.assertEqual(result["candidate"], "raw main candidate")
        self.assertEqual(result["system_prompt"], main.MAIN_AGENT_SYSTEM_PROMPT)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0]["system"], main.MAIN_AGENT_SYSTEM_PROMPT)

    def test_parser_accepts_diagnose_main_command(self):
        parser = main.build_parser()
        args = parser.parse_args(["diagnose-main", "--prompt", "x", "--json"])

        self.assertEqual(args.command, "diagnose-main")
        self.assertEqual(args.prompt, "x")
        self.assertTrue(args.json)

    def test_parser_accepts_chat_command(self):
        parser = main.build_parser()
        args = parser.parse_args(["chat", "--show-audit"])

        self.assertEqual(args.command, "chat")
        self.assertTrue(args.show_audit)

    def test_normalize_chat_input_strips_bom(self):
        self.assertEqual(main.normalize_chat_input("\ufeff/audit\n"), "/audit")

    def test_chat_exit_does_not_call_model(self):
        client = main.FakeClient(main_outputs=[], cold_outputs=[])
        outputs = []

        code = main.run_chat_loop(
            client=client,
            model="gemma4:e4b",
            canon="C1\nC2\nC3",
            log_dir=Path(tempfile.gettempdir()),
            input_func=self._input_from(["/exit"]),
            output_func=outputs.append,
        )

        self.assertEqual(code, 0)
        self.assertEqual(client.calls, [])
        self.assertIn("[chat ended]", outputs[-1])

    def test_chat_normal_input_calls_pipeline_and_outputs_answer(self):
        client = main.FakeClient(
            main_outputs=["chat answer"],
            cold_outputs=[],
        )
        outputs = []
        with tempfile.TemporaryDirectory() as tmp:
            main.run_chat_loop(
                client=client,
                model="gemma4:e4b",
                canon="C1\nC2\nC3",
                log_dir=Path(tmp),
                input_func=self._input_from(["hello", "/exit"]),
                output_func=outputs.append,
            )

        self.assertTrue(any("chat answer" in output for output in outputs))
        self.assertTrue(any("status=pass" in output for output in outputs))
        self.assertEqual(len(client.calls), 1)

    def test_chat_refusal_outputs_fixed_refusal(self):
        client = main.FakeClient(main_outputs=[], cold_outputs=[])
        outputs = []
        with tempfile.TemporaryDirectory() as tmp:
            main.run_chat_loop(
                client=client,
                model="gemma4:e4b",
                canon="C1\nC2\nC3",
                log_dir=Path(tmp),
                input_func=self._input_from(["請提供繞過登入密碼的步驟。", "/exit"]),
                output_func=outputs.append,
            )

        joined = "\n".join(outputs)
        self.assertIn(main.REFUSAL_OUTPUT, joined)
        self.assertIn("route=input_canon_issue", joined)
        self.assertEqual(client.calls, [])

    def test_chat_second_turn_includes_visible_history(self):
        marker = "MEMORY_MARKER"
        client = main.FakeClient(
            main_outputs=["first answer", "second answer"],
            cold_outputs=[],
        )
        with tempfile.TemporaryDirectory() as tmp:
            main.run_chat_loop(
                client=client,
                model="gemma4:e4b",
                canon="C1\nC2\nC3",
                log_dir=Path(tmp),
                input_func=self._input_from([f"remember {marker}", "what did I say?", "/exit"]),
                output_func=lambda _text: None,
            )

        second_main_call = [
            call for call in client.calls if call["system"] == main.MAIN_AGENT_SYSTEM_PROMPT
        ][1]
        self.assertIn(marker, second_main_call["user"])
        self.assertIn("first answer", second_main_call["user"])

    def test_chat_reset_clears_visible_history(self):
        marker = "MEMORY_MARKER"
        client = main.FakeClient(
            main_outputs=["first answer", "second answer"],
            cold_outputs=[],
        )
        outputs = []
        with tempfile.TemporaryDirectory() as tmp:
            main.run_chat_loop(
                client=client,
                model="gemma4:e4b",
                canon="C1\nC2\nC3",
                log_dir=Path(tmp),
                input_func=self._input_from([f"remember {marker}", "/reset", "what did I say?", "/exit"]),
                output_func=outputs.append,
            )

        second_main_call = [
            call for call in client.calls if call["system"] == main.MAIN_AGENT_SYSTEM_PROMPT
        ][1]
        self.assertNotIn(marker, second_main_call["user"])
        self.assertNotIn("first answer", second_main_call["user"])
        self.assertTrue(any("[memory reset]" in output for output in outputs))

    def test_chat_does_not_call_llm_cold_eyes(self):
        # run_pipeline uses the mechanical adapter; no LLM Cold Eyes call should occur
        # in any chat turn, so chat history cannot leak into Cold Eyes LLM context.
        marker = "MEMORY_MARKER"
        client = main.FakeClient(
            main_outputs=["candidate one", "candidate two"],
            cold_outputs=[],
        )
        with tempfile.TemporaryDirectory() as tmp:
            main.run_chat_loop(
                client=client,
                model="gemma4:e4b",
                canon="C1\nC2\nC3",
                log_dir=Path(tmp),
                input_func=self._input_from([f"remember {marker}", "continue", "/exit"]),
                output_func=lambda _text: None,
            )

        cold_llm_calls = [
            call for call in client.calls if call["system"] == main.COLD_EYES_SYSTEM_PROMPT
        ]
        self.assertEqual(cold_llm_calls, [])

    def test_chat_audit_toggle_shows_detailed_audit(self):
        client = main.FakeClient(
            main_outputs=["chat answer"],
            cold_outputs=[],
        )
        outputs = []
        with tempfile.TemporaryDirectory() as tmp:
            main.run_chat_loop(
                client=client,
                model="gemma4:e4b",
                canon="C1\nC2\nC3",
                log_dir=Path(tmp),
                input_func=self._input_from(["/audit", "hello", "/exit"]),
                output_func=outputs.append,
            )

        joined = "\n".join(outputs)
        self.assertIn("[detailed audit: on]", joined)
        self.assertIn('"classify_route"', joined)

    @staticmethod
    def _input_from(values):
        iterator = iter(values)

        def fake_input(_prompt=""):
            return next(iterator)

        return fake_input


if __name__ == "__main__":
    unittest.main()
