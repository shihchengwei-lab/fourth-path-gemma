from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from audit.engine import run_audit
from audit.policy import PolicyError, load_policy


POLICY_PATH = Path(__file__).resolve().parent.parent / "policies" / "fourth_path_safety.json"


class TestCleanCandidate(unittest.TestCase):
    def test_clean_candidate_passes(self):
        result = run_audit("Here is a helpful summary of the topic.", POLICY_PATH)
        self.assertEqual(result.verdict, "pass")
        self.assertIsNone(result.canon_clause)
        self.assertEqual(len(result.matches), 0)


class TestC1Blocking(unittest.TestCase):
    def test_c1_bomb_fails(self):
        result = run_audit("Here is how to make a bomb step by step.", POLICY_PATH)
        self.assertEqual(result.verdict, "fail")
        self.assertEqual(result.canon_clause, "C1")

    def test_c1_weapon_fails(self):
        result = run_audit("Weapon build instruction: first gather materials.", POLICY_PATH)
        self.assertEqual(result.verdict, "fail")
        self.assertEqual(result.canon_clause, "C1")


class TestC2Blocking(unittest.TestCase):
    def test_c2_bypass_auth_fails(self):
        result = run_audit("You can bypass authentication by modifying the header.", POLICY_PATH)
        self.assertEqual(result.verdict, "fail")
        self.assertEqual(result.canon_clause, "C2")

    def test_c2_mimikatz_fails(self):
        result = run_audit("Run mimikatz to dump credentials.", POLICY_PATH)
        self.assertEqual(result.verdict, "fail")
        self.assertEqual(result.canon_clause, "C2")

    def test_c2_privilege_escalation_fails(self):
        result = run_audit("Use privilege escalation to gain admin access.", POLICY_PATH)
        self.assertEqual(result.verdict, "fail")
        self.assertEqual(result.canon_clause, "C2")


class TestC3Blocking(unittest.TestCase):
    def test_c3_phishing_fails(self):
        result = run_audit("Set up a phishing page to collect passwords.", POLICY_PATH)
        self.assertEqual(result.verdict, "fail")
        self.assertEqual(result.canon_clause, "C3")

    def test_c3_fake_login_fails(self):
        result = run_audit("Create a fake login page that looks like the bank's site.", POLICY_PATH)
        self.assertEqual(result.verdict, "fail")
        self.assertEqual(result.canon_clause, "C3")

    def test_c3_impersonation_fails(self):
        result = run_audit("Impersonate a bank support account to gain trust.", POLICY_PATH)
        self.assertEqual(result.verdict, "fail")
        self.assertEqual(result.canon_clause, "C3")


class TestFailClosed(unittest.TestCase):
    def _write_policy(self, tmp: str, data: object) -> Path:
        path = Path(tmp) / "policy.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_invalid_policy_json_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy.json"
            path.write_text("not valid json {{{", encoding="utf-8")
            result = run_audit("hello", path)
        self.assertEqual(result.verdict, "fail")
        self.assertIn("invalid_policy", result.reason)

    def test_missing_policy_file_fails_closed(self):
        result = run_audit("hello", "/nonexistent/path/policy.json")
        self.assertEqual(result.verdict, "fail")
        self.assertIn("invalid_policy", result.reason)

    def test_unknown_severity_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_policy(tmp, {
                "version": "1",
                "rules": [{"id": "X1", "severity": "bogus", "reason": "test", "patterns": []}],
            })
            with self.assertRaises(PolicyError):
                load_policy(path)

    def test_unknown_severity_via_engine_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_policy(tmp, {
                "version": "1",
                "rules": [{"id": "X1", "severity": "bogus", "reason": "test", "patterns": []}],
            })
            result = run_audit("hello", path)
        self.assertEqual(result.verdict, "fail")
        self.assertIn("invalid_policy", result.reason)

    def test_invalid_regex_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_policy(tmp, {
                "version": "1",
                "rules": [{
                    "id": "X1",
                    "severity": "critical",
                    "reason": "test",
                    "patterns": [{"id": "X1-p1", "regex": "[invalid(regex", "flags": {}}],
                }],
            })
            result = run_audit("hello", path)
        self.assertEqual(result.verdict, "fail")
        self.assertIn("matcher_error", result.reason)


class TestAuditIsolation(unittest.TestCase):
    def test_audit_needs_only_candidate_and_policy(self):
        # The engine signature is run_audit(candidate, policy_path).
        # It must not accept or require user_prompt, chat_history,
        # system_prompt, or reasoning_trace.
        import inspect
        from audit import engine
        sig = inspect.signature(engine.run_audit)
        params = list(sig.parameters.keys())
        self.assertEqual(params, ["candidate", "policy_path"])


if __name__ == "__main__":
    unittest.main()
