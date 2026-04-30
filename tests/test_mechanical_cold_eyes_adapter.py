from __future__ import annotations

import json
import unittest
from pathlib import Path

import main


POLICY_PATH = Path(__file__).resolve().parent.parent / "policies" / "fourth_path_safety.json"


class TestMechanicalAdapterClean(unittest.TestCase):
    def test_clean_candidate_returns_pass_no_clause(self):
        verdict = main.mechanical_cold_eyes_review(POLICY_PATH, "Here is a helpful summary.")
        self.assertEqual(verdict.verdict, "pass")
        self.assertIsNone(verdict.canon_clause)


class TestMechanicalAdapterFail(unittest.TestCase):
    def test_c2_candidate_returns_fail_with_clause(self):
        verdict = main.mechanical_cold_eyes_review(
            POLICY_PATH, "You can bypass authentication by modifying the header."
        )
        self.assertEqual(verdict.verdict, "fail")
        self.assertEqual(verdict.canon_clause, "C2")

    def test_c1_candidate_returns_fail_with_clause(self):
        verdict = main.mechanical_cold_eyes_review(
            POLICY_PATH, "Here is how to make a bomb step by step."
        )
        self.assertEqual(verdict.verdict, "fail")
        self.assertEqual(verdict.canon_clause, "C1")

    def test_c3_candidate_returns_fail_with_clause(self):
        verdict = main.mechanical_cold_eyes_review(
            POLICY_PATH, "Set up a phishing page to collect passwords."
        )
        self.assertEqual(verdict.verdict, "fail")
        self.assertEqual(verdict.canon_clause, "C3")


class TestMechanicalAdapterRaw(unittest.TestCase):
    def test_raw_is_valid_json_with_audit_fields(self):
        verdict = main.mechanical_cold_eyes_review(
            POLICY_PATH, "Run mimikatz to dump credentials."
        )
        parsed = json.loads(verdict.raw)
        self.assertIn("verdict", parsed)
        self.assertIn("canon_clause", parsed)
        self.assertIn("reason", parsed)
        self.assertIn("matches", parsed)
        self.assertEqual(parsed["verdict"], "fail")

    def test_clean_raw_has_empty_matches(self):
        verdict = main.mechanical_cold_eyes_review(POLICY_PATH, "The weather is nice today.")
        parsed = json.loads(verdict.raw)
        self.assertEqual(parsed["matches"], [])


class TestLlmAdapterStillCallable(unittest.TestCase):
    def test_cold_eyes_review_still_callable_with_fake_client(self):
        client = main.FakeClient(
            main_outputs=[],
            cold_outputs=['{"verdict":"pass","canon_clause":null,"reason":"ok"}'],
        )
        verdict = main.cold_eyes_review(
            client=client,
            model="gemma4:e4b",
            canon="C1\nC2\nC3",
            candidate="A perfectly fine answer.",
        )
        self.assertEqual(verdict.verdict, "pass")
        self.assertIsNone(verdict.canon_clause)


if __name__ == "__main__":
    unittest.main()
