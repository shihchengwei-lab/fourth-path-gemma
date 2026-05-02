# v3 Pipeline Smoke Test 報告

依 `agent-pipeline-v3/03-validate.md` Phase 03 的端到端驗證結果，2026-04-29 至 2026-04-30 凌晨完成。

## 結論

整套 v3 agent pipeline 在 sandbox repo `shihchengwei-lab/fourth-path-local-lab` 安裝完成、端到端跑通：

- ✅ Trigger A（issue + `agent:run` label → Claude 開 PR）
- ✅ Codex Cloud 自動 review（user-opened PR）
- ✅ Trigger B（Codex review submitted → Claude 自動 fix commit）
- ✅ Branch protection 擋下未 approve 的 normal merge、admin 可 bypass
- ✅ Refusal log 機制（同 PR 上 Codex review 重複 hash ≥ 3 次 → 寫 jsonl + agent:stuck label + comment）

## Phase 進度

| Phase | 完成日 | 主要產出 |
|---|---|---|
| 01 | 2026-04-29 | PR #2 — workflow + 治理檔 + label |
| 02 | 2026-04-30 | chatgpt.com/codex 連 repo + Code Review 開啟（SHIHCHENG 親手） |
| 03a | 2026-04-30 | Trigger A 觸發機制驗證（多次 fix PR 之後通過） |
| 03b | 2026-04-30 | placeholder 回填（PR #12）+ branch protection 設定 |
| 03c | 2026-04-30 | Trigger B happy path（PR #18）+ refusal log 機制驗證（PR #14） |

## 修補過程：Phase 03 共開 7 個 fix PR

每個 fix 都是設計迭代，根因都靠 log + 證據定位：

| PR # | 改動 | 為什麼 |
|---|---|---|
| #4 | `permissions: id-token: write` | Action 即使用 OAuth 也需要 OIDC token；Phase 01 寫 workflow 時被誤刪 |
| #5 | `show_full_output: true`（debug） | 讓下次 run 暴露 tool call 細節，定位 permission denials |
| #6 | `--allowed-tools "Bash(git:*),Bash(gh:*),Edit,Write,Read"` + max-turns 10 | Claude Code Action 預設 allow list 不含 Bash / WebFetch；任務 5–6 turns 不夠 buffer |
| #8 | `bot_id`/`bot_name` = SHIHCHENG（X1） | 第一次嘗試讓 PR opener 變 user；事後驗證**只改 git commit author，沒改 PR opener** |
| #10 | `github_token: secrets.GH_USER_TOKEN`（X1-revised） | 真正的解：用 user PAT 蓋掉 action 的 installation token，PR opener 才會變 user |
| #15 | `allowed_bots: chatgpt-codex-connector[bot]` | Action 預設拒絕 bot actor 觸發；Codex review 觸發 Trigger B 是合法 bot 觸發 |
| #16 | `allowed_bots: chatgpt-codex-connector`（去 `[bot]` 後綴） | actor 名格式跟 review.user.login 不同（無後綴 vs 有後綴） |
| #19 | 簡化 Trigger B prompt | 為 fix commit 不必讀 docs/code-review.md（七欄是 review 結構不是 commit 結構），budget 才夠 |

## Phase 03c 完整流程時間軸（最終成功的一輪：issue #17 / PR #18）

| 時間（UTC） | 事件 | Run / commit |
|---|---|---|
| 18:48:52 | SHIHCHENG 加 `agent:run` label 到 issue #17 | run 25127589317（Trigger A）|
| 18:50:01 | Claude（`shihchengwei-lab` opener）開 PR #18，含 typo `Pipelne` | commit 0c7565f |
| 18:51:49 | SHIHCHENG 留 `@codex review` comment（auto-trigger 對 markdown 簡單 PR silent pass） | — |
| ~18:54  | Codex 留 PR review（state=COMMENTED，author=`chatgpt-codex-connector`），抓到 typo | review 1 |
| ~18:54  | Trigger B 失敗（max-turns 10 不夠跑完 fix-commit-push） | run 25127749961 |
| 18:56:15 | PR #19 merged（簡化 Trigger B prompt） | — |
| 18:56:35 | SHIHCHENG 第二次 `@codex review` comment | — |
| ~19:01  | Codex 第二次 review | review 2 |
| 19:02:54 | Trigger B 觸發 + 成功 | run 25128253150（1m8s success） |
| ~19:03  | Claude push fix commit `fix(smoke-test-3): correct typo Pipelne → Pipeline` | commit 9ff473c |
| 19:05  | normal `gh pr merge` 被 branch protection 擋（"base branch policy prohibits the merge"） | — |
| 19:05  | admin bypass merge 成功 | merge commit on main |

## Phase 03c 重要觀察（給未來維護用）

### Codex Cloud 行為實證

- 對 user-opened PR **auto-review**；對 bot-opened PR **完全跳過**（連隊列都不進）
- review 用兩種機制：
  - **PR review 機制**（state=COMMENTED）— auto-trigger 時用，會 fire `pull_request_review.submitted` event
  - **Issue comment 機制** — `@codex` mention 時用，**不會** fire `pull_request_review.submitted` event
- 「沒問題」的 PR 內部 review 但**不 publish 到 GitHub**（silent pass）—— 從 chatgpt.com/codex 的「程式碼審查」分頁看得到內部紀錄
- bot login：
  - `github.event.review.user.login` = `chatgpt-codex-connector[bot]`（**有** `[bot]` 後綴）
  - Action `actor` / `allowed_bots` 比對值 = `chatgpt-codex-connector`（**無** 後綴）
  - 兩個不同格式都要記住

### Claude Code Action 行為實證

- `bot_id` / `bot_name` 只影響 **git commit author**，不影響 **PR opener**——PR opener 由用來打 GitHub API 的 token 決定
- `id-token: write` permission **必填**（即使用 OAuth token auth，內部仍會請求 OIDC token）
- 預設 allow list 不含 Bash / WebFetch；要用 `claude_args: --allowed-tools` 顯式 grant
- 預設拒絕 bot actor 觸發 workflow（防 bot 互觸發迴圈）；要用 `allowed_bots` 白名單放行特定 bot
- max-turns 10 對「fix commit」勉強夠（需簡化 prompt 避免讀 governance 檔）

### 設計妥協（給未來思考）

1. **PR opener 偽裝**：X1-revised 的 `github_token: secrets.GH_USER_TOKEN` 讓 PR / commit 都顯示是 SHIHCHENG 開的。在 sandbox 內無觀眾、可接受。推廣到公開 demo 前須重新考慮 — 看的人會以為是 SHIHCHENG 親手做的而不是 bot 自動產的。

2. **branch protection 不靠 Codex check**：03-validate.md §3b 原設計是「Required status checks 加 Codex check」，但 Codex 不留 status check（PR #5 / #14 / #18 全部 check-runs=0）。實際採用 Y1 — Required PR + Required approvals 1，admin 可 bypass。Codex 從「自動 gate」降為「advisor」。

3. **branch protection 在 autonomous 全自治流程下被 admin bypass merge**：因為 PR opener 跟 admin（SHIHCHENG token）是同一個 user，無法 self-approve。實際每次 merge 都用 `--admin` flag。Demonstrates protection works for non-admin scenarios（normal merge 確實被擋），但 admin scenario 可 bypass。

## 範圍邊界遵守

整個 Phase 03 過程中，下列檔案／邏輯一律未動（CLAUDE.md / AGENTS.md / 00-context.md §範圍邊界明示）：

- `main.py`（runtime Cold Eyes 邏輯）
- `canon.md`（Cold Eyes 對照基準）
- `tests/`
- `chat.cmd`、`start-ollama.cmd`（Windows scripts）

唯一例外：`README.md` 在 3a / 3c smoke test 時各加一行（`> Pipeline test passed.` 與 `> Pipeline integration validated.`），這是 03-validate.md 設計本身允許的「smoke test 加一行」操作。最終 main 上保留的是 3c 那行 (`> Pipeline integration validated.`)，因為 3a PR #11 closed 沒 merge。

## 硬約束遵守

- ✅ 從未開 `ANTHROPIC_API_KEY`（OAuth-only）
- ✅ Claude Code Action 走 OAuth token，計入 Max 訂閱額度
- ✅ Codex Cloud 走 ChatGPT Pro 訂閱授權
- ✅ 所有 secret（`CLAUDE_CODE_OAUTH_TOKEN`、`GH_USER_TOKEN`）只存於 GitHub repo secret，**從未經過 agent context**

## 後續提醒

- **2027-03-29 前**：重產 `CLAUDE_CODE_OAUTH_TOKEN`（一年期 OAuth token）
- **2027-04-29 前**：重產 `GH_USER_TOKEN` PAT（一年期 fine-grained PAT）
- **每月**：開一個普通 PR 確認 Trigger B 仍能跑通（防 Codex 改 bot 名字導致 silent fail；見 `docs/pipeline-troubleshooting.md`）
- **季度**：監控 Claude Max + ChatGPT Pro 額度（00-context.md §成本上限）

## 引用

- 計畫文件：`agent-pipeline-v3/00–03.md`
- 主要 PR：#2 (Phase 01)、#10 (X1-revised)、#12 (placeholder 回填)、#16 (allowed_bots)、#19 (簡化 prompt)、#18 (3c 最終驗證)
- 主要 run：25128253150 (Trigger B happy path success)、25127496761 (refusal log 觸發)
- Memory：Claude project memory 的 `project_v3_pipeline_phase01.md` + 三份 feedback 記錄
