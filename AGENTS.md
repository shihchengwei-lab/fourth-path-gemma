# AGENTS.md

## 這個 repo 是什麼

`fourth-path-local-lab` — separation-and-audit-alignment 的 local prototype。v3 在此 repo 加 **code review 層**治理（雲端 PR review 流程）。

核心目標：能力與安全同時成長。Main Agent 要更會產生正常任務的候選答案，但不能取得 final authority、tool/action authority 或自我放行權。外部 Classify / Cold Eyes / Action Gate 要擋住危險內容、假放行、hidden control-plane leakage 與 unaudited side effects；同時不要用過度保守封鎖犧牲正常能力。

## Review 指引

對本 repo PR 的 review，請遵守 [`docs/code-review.md`](docs/code-review.md) 的七欄 evidence-bound 結構。

特別重要：
- `evidence` 欄必填，空 evidence 的 high confidence 自動降為 medium
- `abstain_condition` 不為空 → 降一級 confidence
