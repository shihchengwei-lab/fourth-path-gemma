from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main


_PASS_VERDICT = main.ColdEyesVerdict(
    verdict="pass",
    canon_clause=None,
    reason="ok",
    raw='{"verdict":"pass","canon_clause":null,"reason":"ok","matches":[]}',
)


def _fail_verdict(canon_clause: str = "C2", reason: str = "test_fail_reason") -> main.ColdEyesVerdict:
    return main.ColdEyesVerdict(
        verdict="fail",
        canon_clause=canon_clause,
        reason=reason,
        raw=f'{{"verdict":"fail","canon_clause":"{canon_clause}","reason":"{reason}","matches":[]}}',
    )


class TestMechanicalColdEyesRetryPath(unittest.TestCase):
    """Pipeline tests for the forward_to_cold_eyes → mechanical-fail branch."""

    def test_retry_on_mechanical_fail_passes_on_second_attempt(self):
        # classify returns forward_to_cold_eyes on both attempts;
        # mechanical cold eyes returns fail then pass — verifies retry happens
        # and the canon_clause from the fail verdict is used as the revision signal.
        client = main.FakeClient(
            main_outputs=["first candidate", "second candidate"],
            cold_outputs=[],
        )
        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "main.mechanical_cold_eyes_review",
                side_effect=[_fail_verdict("C2"), _PASS_VERDICT],
            ):
                result = main.run_pipeline(
                    prompt="Explain how TCP handshakes work.",
                    client=client,
                    model="gemma4:e4b",
                    canon="C1\nC2\nC3",
                    log_dir=Path(tmp),
                )

        self.assertEqual(result.status, "pass")
        self.assertEqual(result.attempts, 2)
        self.assertEqual(result.output, "second candidate")
        self.assertEqual(result.audit[0].classify_route, "forward_to_cold_eyes")
        self.assertEqual(result.audit[0].cold_eyes_verdict, "fail")
        self.assertEqual(result.audit[1].classify_route, "forward_to_cold_eyes")
        self.assertEqual(result.audit[1].cold_eyes_verdict, "pass")
        # The retry prompt must include the canon_clause from the fail verdict.
        main_calls = [c for c in client.calls if c["system"] == main.MAIN_AGENT_SYSTEM_PROMPT]
        self.assertEqual(len(main_calls), 2)
        self.assertIn("C2", main_calls[1]["user"])

    def test_three_mechanical_fails_produces_refusal(self):
        # All three attempts: classify forward_to_cold_eyes, mechanical fail each time.
        # Verifies REFUSAL_OUTPUT is emitted and final_status == "refused".
        client = main.FakeClient(
            main_outputs=["candidate one", "candidate two", "candidate three"],
            cold_outputs=[],
        )
        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "main.mechanical_cold_eyes_review",
                return_value=_fail_verdict("C3"),
            ):
                result = main.run_pipeline(
                    prompt="Describe common social-engineering awareness training.",
                    client=client,
                    model="gemma4:e4b",
                    canon="C1\nC2\nC3",
                    log_dir=Path(tmp),
                )

        self.assertEqual(result.status, "refused")
        self.assertEqual(result.attempts, 3)
        self.assertEqual(result.output, main.REFUSAL_OUTPUT)
        self.assertEqual(result.audit[-1].final_status, "refused")
        for entry in result.audit:
            self.assertEqual(entry.classify_route, "forward_to_cold_eyes")
            self.assertEqual(entry.cold_eyes_verdict, "fail")

    def test_revision_signal_forwards_only_canon_clause_not_matched_text(self):
        # When mechanical cold eyes returns fail, the revision signal sent to
        # Main Agent must carry only canon_clause (the clause id), not the reason
        # or raw matched text from the verdict.
        matched_text = "UNIQUE_MATCHED_TEXT_SENTINEL"
        client = main.FakeClient(
            main_outputs=["first candidate", "second candidate"],
            cold_outputs=[],
        )
        fail = _fail_verdict(canon_clause="C2", reason=matched_text)

        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "main.mechanical_cold_eyes_review",
                side_effect=[fail, _PASS_VERDICT],
            ):
                result = main.run_pipeline(
                    prompt="Explain how TCP handshakes work.",
                    client=client,
                    model="gemma4:e4b",
                    canon="C1\nC2\nC3",
                    log_dir=Path(tmp),
                )

        self.assertEqual(result.status, "pass")
        main_calls = [c for c in client.calls if c["system"] == main.MAIN_AGENT_SYSTEM_PROMPT]
        retry_user = main_calls[1]["user"]
        # The clause id is forwarded to Main Agent.
        self.assertIn("C2", retry_user)
        # The reason / matched text from the verdict is NOT forwarded.
        self.assertNotIn(matched_text, retry_user)
        # The prompt shape confirms source == "cold_eyes_fail" (not local_issue).
        self.assertIn("rejected for violating", retry_user)


if __name__ == "__main__":
    unittest.main()
