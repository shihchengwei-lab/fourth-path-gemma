# Refusal Log 規範

## 觸發

Codex 對同一 PR 的 review 內容（hash 相同）連續出現 3 次 → workflow 自動觸發。

## 寫入位置

`docs/refusal-log.jsonl`，每行一條 JSON。

## 格式

```json
{
  "timestamp": "ISO8601 UTC",
  "task_id": "<branch name or HEAD sha>",
  "output_artifact": {
    "diff_hash": "sha256 of the failing PR diff",
    "files": ["..."]
  },
  "issue_signatures": ["..."],
  "retry_count": 3,
  "user_decision": null,
  "user_decision_notes": null,
  "resolved_at": null
}
```

## SHIHCHENG 裁決流程

1. 看 PR 上的 `agent:stuck` label
2. 判斷該 issue 是否：
   - 應該繼續推進（修方向）
   - 應該關閉（不適合做）
   - 需要拆得更小（重開）
3. 手動更新 `refusal-log.jsonl` 那條記錄的 `user_decision` / `user_decision_notes` / `resolved_at`
