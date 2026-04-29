# CLAUDE.md

## 這個 repo 是什麼

`fourth-path-gemma` — [separation-and-audit-alignment](https://github.com/shihchengwei-lab/separation-and-audit-alignment) 的 local prototype（本機 LLM + canon.md + Cold Eyes runtime 層）。

v3 在此 repo 加 **code review 層**治理。

## 範圍邊界

- **不要動**：`main.py`、`canon.md`、`tests/`、Windows script、runtime Cold Eyes 邏輯
- **只新增**：`.github/workflows/claude.yml`、`docs/code-review.md`、`docs/refusal-log.md`、`docs/pipeline-troubleshooting.md`、本檔（`CLAUDE.md`）、`AGENTS.md`
- 唯一例外：`README.md` 在 Phase 03 smoke test 時會加一行（測試本身的設計）

## Review 風格

對本 repo 的所有 review、self-review、PR 描述，請遵守 [`docs/code-review.md`](docs/code-review.md) 的七欄 evidence-bound 結構。

## 開 PR 時

PR description 中的問題列表也應使用七欄結構（至少含 `severity`、`evidence`、`suggested_validation` 三欄，其餘可省略）。
