# msb-sre-agent

AgentBase agent on GreenNode AgentBase — standardized by the `agentbase-build` skill.

| Component | Technology |
|---|---|
| Backend | Python 3.13 · uv · LangGraph · `greennode-agentbase` (`src/backend`) |
| LLM | GreenNode AI Platform MaaS (OpenAI-compatible) |
| Short-term memory | AgentBase Memory — `AgentBaseMemoryEvents` checkpointer |
| Long-term memory | AgentBase Memory records — auto-recall + tools `remember`/`recall_memory` |
| Context compression | Rolling summary + hard trim |
| Tools | Local tools + MCP via AgentBase MCP Gateway (Connector + Policy Group) |
| Auth | JWT (OIDC/JWKS) inbound · AgentBase Identity outbound |
| Tracing | Langfuse |
| Frontend | Expo React Native (`src/frontend`, optional) |

## Run locally

```bash
make setup
cp src/backend/.env.example src/backend/.env   # fill in LLM_API_KEY, LLM_MODEL
make test
make dev
make invoke MSG="hello"
```

## Architecture & conventions

See `src/backend/app/*` — each module has a docstring describing its responsibility. API contract: `src/backend/app/service.py`.

## Deploy

`make docker-build`, then use `/agentbase-deploy` (push to AgentBase Container Registry → create a Runtime with env file `src/backend/.env.<env>`).

---

## msb-sre-agent — vận hành

Trợ lý SRE điều tra sự cố trên dữ liệu Elasticsearch on-prem (logs + metrics). Spec: `docs/agent-spec.md`. Trạng thái tài nguyên: `.agentbase-state.json`.

| Thành phần | Giá trị |
|---|---|
| Runtime | `msb-sre-agent` (`runtime-843199e5-…`), Public, `runtime-s2-general-2x4`, 1 replica |
| Endpoint | `https://endpoint-9c5d7eee-5208-4dc9-a1e0-dcf50d8b9403.agentbase-runtime.aiplatform.vngcloud.vn` (`/health`, `/invocations`, `/webhook/zalo`) |
| LLM | `z-ai/glm-5.3-flash-thirdparty`, fallback `qwen/qwen3.8-flash`, key `msb-sre-agent-key` |
| MCP | Gateway `msb-samples-mcp-gateway` → connector `msb_elasticsearch_onpermise` (tool phía agent: `es_<tool>`) |
| Policy | `msb-samples-mcp-policy`: `allow-msb-sre-agent-dev` (chạy local/eval) và `allow-msb-sre-agent-runtime` |
| Secret | LLM key + hash API key trong `src/backend/.env.prod`; token Zalo + webhook secret trong Access Control |

Lệnh hay dùng (trong thư mục này, cần `export PATH=$HOME/.local/bin:$PATH`):

- `make test`, `make lint`, `make eval` (eval chạy ~15 phút vì MaaS giới hạn 10 request/phút).
- `make zalo-webhook ARGS=me` kiểm tra token bot; `make zalo-webhook ARGS="set --url <endpoint>/webhook/zalo"` đặt webhook.
- Thêm người dùng Zalo: sửa `ZALO_ALLOWED_USER_IDS=["<zalo-id>", ...]` trong `src/backend/.env.prod`, rồi `runtime.sh update <runtime-id> --image <image> --flavor runtime-s2-general-2x4 --env-file src/backend/.env.prod --min-replicas 1 --max-replicas 1 --cpu-scale 50 --mem-scale 50 --from-cr` (skill agentbase-deploy). Người lạ nhắn bot sẽ nhận Zalo ID của họ.
- Đổi tên connector trên Console thì phải sửa URL trong `src/backend/mcp_servers.json` và action của policy (`<connector>__<tool>`).
- Log không chứa token bot: `RedactBotToken` (`src/backend/app/channels/zalo.py`) che đoạn `/bot<token>` trong log của httpx (`https://bot-api.zaloplatforms.com/bot<redacted>/sendMessage`); test ở nhóm `log hygiene` trong `tests/test_zalo.py`. Log cũ của các phiên bản đã chạy trước khi có bản vá vẫn còn token, nên token đó phải được đổi.
- Đổi token Zalo (bị lộ hoặc định kỳ): lấy token mới ở Zalo Bot Creator → cập nhật provider `msb-sre-zalo-bot-token` trong Access Control → `make zalo-webhook ARGS=me` để kiểm tra → `make zalo-webhook ARGS="set --url <endpoint>/webhook/zalo"` để đặt lại webhook. Runtime đọc lại token sau tối đa `ZALO_SECRET_TTL_S` giây, không cần deploy lại.

Chưa làm: tracing Langfuse (cần `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`), Rate Limit cho LLM key (Console → Protect & Govern).
