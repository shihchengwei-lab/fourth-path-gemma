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
