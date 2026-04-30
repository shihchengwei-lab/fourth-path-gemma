# Pipeline Troubleshooting

## 1. Claude 不再對 Codex review 自動回應

**症狀**：Codex 在 PR 貼了 review，但 Claude Code Action 沒有被觸發、PR 上沒有新 commit。

**檢查順序**：

1. 進 `.github/workflows/claude.yml`，找到 Trigger B 的 `if:` 條件
2. 看條件裡寫死的 reviewer login 字串
3. 對照最近一個真實 Codex review 的 author login（PR → review → 滑鼠移到 reviewer 頭像）
4. 不一致 → Codex 改名了，更新 workflow 的字串
5. 一致但仍不觸發 → 看 Actions tab 是否有錯誤 log

**預防**：每月開一個普通測試 PR，確認 pipeline 仍能跑通。

## 2. Codex 回覆「create an environment for this repo」

**症狀**：PR 上的 `@codex review` 沒有產生 PR review，反而收到 Codex comment：

```text
To use Codex here, create an environment for this repo.
```

**判斷**：這不是 workflow YAML 或 Claude 修正邏輯失敗，而是 ChatGPT/Codex Cloud 的 repo environment 尚未啟用或已失效。workflow 無法自行建立這個外部帳號設定。

**處理**：

1. 到 Codex Cloud settings 建立或修復此 repo 的 environment。
2. 回到 PR 留一次 `@codex review`。
3. 若 Codex 產生 PR review，Trigger B 會繼續自動修正並要求下一輪 review。

**自動標記**：workflow 會在偵測到這個 Codex comment 時，對該 PR 加上 `agent:needs-user-decision`，並只留一則說明 comment。

## 3. Claude 修了 workflow 但 push 被擋

**症狀**：Claude Code Action 已經在 runner 裡 commit 了 `.github/workflows/*` 變更，但 push 失敗，訊息提到 token 需要 `workflow` scope。

**判斷**：這是 GitHub token 權限問題。只要修正內容包含 workflow 檔案，推送 token 必須有 `workflow` scope；否則 Trigger B 可以讀 review、可以本地 commit，但不能把修正推回 PR branch。

**處理**：

1. 重新產生或更新 repo secret `GH_USER_TOKEN`，確保包含 `workflow` scope。
2. 重新觸發該 PR 的 review/fix 流程，或由本機用具備 `workflow` scope 的 git credential 推上同一個 branch。
3. 修復後再留一次 `@codex review`，讓 Codex 重新審最新 commit。
