# Code Review 風格指引（evidence-bound 七欄 PR review 結構）

任何對本 repo PR 的 review（Codex / Claude self / 人類）都應遵守此結構。

> 本 repo 的 `main.py` 已有 runtime 層的 Cold Eyes（審 LLM output vs canon.md）。本檔規範的是 **code review 層**，不是 runtime 層。兩者層級不同，並存無衝突。

## 七欄結構

| 欄位 | 用途 |
|---|---|
| `severity` | `critical` / `major` / `minor` |
| `confidence` | `high` / `medium` / `low` |
| `category` | `reference` / `security` / `correctness` / `style` / ... |
| `evidence` | **必填**——具體 diff 行或事實。空 evidence 的 high confidence **自動降為 medium** |
| `what_would_falsify_this` | 此 claim 在什麼條件下不成立 |
| `suggested_validation` | 怎麼驗證（跑哪個測試 / 看哪個檔案） |
| `abstain_condition` | 此 claim 隱含哪些未驗證假設。**有 abstain 條件就降一級 confidence** |

## severity 定義

- `critical` — production crash / data loss / security breach / 證據確鑿的 correctness bug
- `major` — 正常使用下行為錯誤
- `minor` — 不理想但可運作

## 來源

此規格搬自 cold-eyes-reviewer 工具（本機 `C:\Users\kk789\Desktop\cold-eyes-reviewer`）的 `cold-review-prompt.txt`。**搬概念不搬程式**——理由見 `agent-pipeline-v3/00-context.md` §Cold Eyes 路徑 C 決議。
