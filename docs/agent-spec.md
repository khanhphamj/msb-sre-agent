# Agent Spec — msb-sre-agent

Đã được người dùng xác nhận ngày 2026-10-07 (Bước 1/9 của agentbase-build).

## 1. Mục tiêu
- Vấn đề: kỹ sư vận hành cần điều tra sự cố nhanh trên dữ liệu Elasticsearch on-prem (logs + metrics của web-01, db-01).
- Người dùng: SRE / vận hành nội bộ MSB (ít người).
- Chỉ số thành công: nhận đúng 4 kịch bản mẫu (cpu_spike, mem_leak, disk_fill, network_spike) — host, khung giờ, nguyên nhân — chỉ bằng tool, không dùng nhãn incident_id.

## 2. Kênh & UI
- Zalo Bot (webhook `POST /webhook/zalo`, chat 1-1) và `POST /invocations` (test / eval). Không có UI web. Streaming: không.

## 3. Authentication
- `/invocations`: `AUTH_MODE=api_key` (không có IdP; IAM không có JWKS). Header `X-GreenNode-AgentBase-Custom-Api-Key`.
- Webhook Zalo: header `X-Bot-Api-Secret-Token` (secret trong Access Control).
- Ai được chat qua Zalo: allow-list Zalo user id (`ZALO_ALLOWED_USER_IDS`); người lạ nhận ID của họ để admin thêm.

## 4. LLM
- Mọi tier: `z-ai/glm-5.3-flash-thirdparty`. Fallback (họ model khác): `qwen/qwen3.8-flash`.
- Adaptive routing: tắt. Ngôn ngữ trả lời: tiếng Việt.
- Giới hạn MaaS: 10 request/phút cho cả tài khoản (fallback không vượt được).

## 5. Tools
| Tool | Nguồn | Đọc/Ghi | HITL | Credential |
|---|---|---|---|---|
| get_overview, search_logs, get_log_stats, get_metrics, get_metrics_summary | MCP qua Gateway `msb-samples-mcp-gateway`, connector `msb_elasticsearch_onpermise` (tên người dùng đặt trên Console) | Đọc | Không | Gateway → MCP server dùng API key trong Access Control (`msb-es-mcp-apikey`) |
- Tên tool phía agent: `es_<tool>` (khóa `es` trong `mcp_servers.json`); action policy: `msb_elasticsearch_onpermise__<tool>`.
- Không có tool cục bộ. Policy Group `msb-samples-mcp-policy`: ALLOW 5 tool cho principal của agent (dev + runtime).

## 6. Memory
- Short-term: bật (memory store `msb-sre-agent-memory`, giữ 30 ngày). Long-term: tắt (không cần cá nhân hóa).

## 7. Human-in-the-loop
- Không (mọi tool chỉ đọc).

## 8. Evaluation
- `evals/datasets/sre.jsonl` (4 incident + overview + cửa sổ thời gian + host lạ + từ chối ghi + ngoài phạm vi + prompt injection). Ngưỡng pass_rate ≥ 0.75. Self-eval: tắt.

## 9. Quyết định thành phần
| Thành phần | Bật? | Lý do |
|---|---|---|
| Long-term memory | Không | Không cần nhớ người dùng qua phiên |
| Reflection | Không | Judge tốn thêm 1–2 call LLM trong khi MaaS giới hạn 10 RPM |
| HITL | Không | Chỉ đọc |
| Build MCP server | Không | Đã có `msb-elasticsearch-mcp` |
| A2A | Không | Không có agent khác |
| Outbound credentials | Có | Token Zalo + webhook secret: Static API Key trong Access Control |
| Private network | Không | Runtime/Gateway Public; Zalo cần gọi vào endpoint công khai |
| Frontend | Không | Kênh là Zalo + API |
| MAX_TOOL_ROUNDS | 8 | Mặc định; điều tra nhiều bước nhưng bị giới hạn 10 RPM |

## 10. Non-functional
- Runtime Public, 1 replica (dedupe/queue của Zalo nằm trong bộ nhớ). Mask PII trong trace. Không log token (URL Zalo chứa token).
