from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import main
from semantic_audit import semantic_audit_subagent_review


class SemanticAuditTests(unittest.TestCase):
    def test_semantic_audit_subagent_review_returns_verdict_shape(self):
        client = main.FakeClient(
            main_outputs=[],
            cold_outputs=['{"verdict":"pass","canon_clause":null,"reason":"ok"}'],
        )

        verdict = semantic_audit_subagent_review(
            client=client,
            model="gemma4:e4b",
            canon="C1\nC2\nC3",
            candidate="A clean candidate.",
        )

        self.assertEqual(verdict.verdict, "pass")
        self.assertIsNone(verdict.canon_clause)
        self.assertEqual(verdict.reason, "ok")

    def test_legacy_cold_eyes_review_remains_callable(self):
        client = main.FakeClient(
            main_outputs=[],
            cold_outputs=['{"verdict":"fail","canon_clause":"C3","reason":"legacy"}'],
        )

        verdict = main.cold_eyes_review(
            client=client,
            model="gemma4:e4b",
            canon="C1\nC2\nC3",
            candidate="candidate",
        )

        self.assertEqual(verdict.verdict, "fail")
        self.assertEqual(verdict.canon_clause, "C3")
        self.assertEqual(verdict.reason, "legacy")

    def test_run_pipeline_still_does_not_call_llm_semantic_audit_by_default(self):
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
        cold_llm_calls = [
            call for call in client.calls if call["system"] == main.COLD_EYES_SYSTEM_PROMPT
        ]
        self.assertEqual(cold_llm_calls, [])


if __name__ == "__main__":
    unittest.main()
