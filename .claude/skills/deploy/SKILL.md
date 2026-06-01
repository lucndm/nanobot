---
name: deploy
description: "Deploy nanobot to production (unraid) via Komodo GitOps. Build image from source, push to Gitea registry, deploy container. Triggers on: 'deploy nanobot', 'push to production', 'release nanobot', 'build nanobot image'."
---

# Nanobot Deploy Skill

Deploy nanobot gateway to production on unraid via Komodo GitOps pipeline.

## Architecture Overview

```
nanobot repo (source code, forked)
  → Phase 1: docker build local → save tar → scp to unraid → docker load → restart
  → Phase 2: git push → Komodo Build → Gitea registry → Komodo Stack deploy
```

Two repos involved:

| Repo | Path | Role |
|------|------|------|
| `nanobot` (this repo) | `/workspaces/github.com/minhluc-info/nanobot` | Source code development |
| `docker-compose` | `/workspaces/github.com/minhluc-info/docker-compose` | Deployment config, Dockerfile, compose, Komodo TOML |

## Deployment Files in docker-compose Repo

| File | Purpose |
|------|---------|
| `stacks/llm/nanobot/nanobot-source/` | **Gitignored** copy of nanobot source, used by Dockerfile |
| `stacks/llm/nanobot/Dockerfile` | Builds image: Python 3.13 + nanobot from local source |
| `stacks/llm/nanobot/compose.yaml` | Container definition: ports, volumes, env, limits |
| `stacks/llm/nanobot/config/config.json` | Config template with `__PLACEHOLDER__` for secrets |
| `komodo/builds/nanobot.toml` | Komodo build config (image registry, tagging) |
| `komodo/stacks/llm/nanobot.toml` | Komodo stack config (server, pre_deploy, secrets) |

## Debug & Test

### Health Check

Gateway chỉ expose 1 HTTP endpoint trên port 18790:

```bash
ssh root@100.68.251.84 'curl -s http://localhost:18790/health'
# {"status":"ok"}
```

Nếu trả `ok` → gateway process đang chạy. Nếu không phản hồi → container crash hoặc port chưa bind.

### Test qua Telegram

Gửi tin nhắn trực tiếp cho bot trên Telegram. Kiểm tra phản hồi:

```bash
ssh root@100.68.251.84 'docker logs nanobot-gateway --tail 30 -f'
```

### Test qua WebUI (WebSocket)

WebUI chạy trên port 8765. Mở browser:

```
http://100.68.251.84:8765
```

WebUI kết nối WebSocket để chat trực tiếp với nanobot. Token auth tự động qua `tokenIssueSecret` trong config.

### Test qua WebSocket CLI (Verified Working)

WebSocket cần auth token. Protocol: GET token → connect WS → nhận `ready` → gửi text → nhận streaming events.

```bash
# One-liner: full WebSocket chat test
timeout 60 ssh root@100.68.251.84 'docker exec nanobot-gateway timeout 45 python3 -c "
import asyncio, json, urllib.request
from websockets.asyncio.client import connect

# 1. Get auth token (secret from .env)
req = urllib.request.Request(\"http://localhost:8765/auth/token\")
req.add_header(\"Authorization\", \"Bearer 0afb95515cc60ef3c290c73b8f0fd6746129ce2a3f2b6460463d83ef31e26b6b\")
token = json.loads(urllib.request.urlopen(req, timeout=5).read())[\"token\"]
print(f\"Token: {token[:20]}...\", flush=True)

async def test():
    # 2. Connect WebSocket (path = \"/\" by default)
    async with connect(f\"ws://localhost:8765/?token={token}\") as ws:
        # 3. Receive ready event
        ready = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
        print(f\"ready: chat_id={ready[\"chat_id\"]}\", flush=True)

        # 4. Send message
        await ws.send(\"say hello in 3 words\")
        print(\"sent: say hello in 3 words\", flush=True)

        # 5. Collect streaming response
        for i in range(40):
            r = await asyncio.wait_for(ws.recv(), timeout=15)
            d = json.loads(r)
            e = d.get(\"event\")
            if e == \"text_delta\" or e == \"delta\":
                print(d.get(\"text\", \"\"), end=\"\", flush=True)
            elif e == \"text_done\" or e == \"stream_end\":
                pass  # stream complete
            elif e == \"turn_end\":
                print(f\"\\n--- done ({d.get(\"latency_ms\")}ms) ---\", flush=True)
                break
            elif e == \"goal_status\":
                print(f\"goal: {d.get(\"status\")}\", flush=True)

asyncio.run(test())
"'
```

### WebSocket Test từ Dev Machine (Local Python)

Khi cần test streaming output chi tiết (check leak, debug tool calls), chạy trực tiếp từ dev machine:

```bash
# WS leak check: gửi message → check XML artifacts trong streaming response
cat << 'PYEOF' | python3 -u -
import asyncio, websockets, json, time, urllib.request

# Get fresh token (TTL 5 min)
req = urllib.request.Request("http://100.68.251.84:8765/auth/token", headers={
    "Authorization": "Bearer 0afb95515cc60ef3c290c73b8f0fd6746129ce2a3f2b6460463d83ef31e26b6b"
})
token = json.loads(urllib.request.urlopen(req).read())["token"]

LEAKS = ["<function", "function=", "<parameter", "</function>", "<tool_call", "</tool_call"]

async def test():
    async with websockets.connect(f"ws://100.68.251.84:8765/?token={token}", ping_interval=30) as ws:
        r = json.loads(await ws.recv())
        chat_id = r.get("chat_id")
        print(f"Connected: {chat_id[:8]}...")

        # === THAY MESSAGE CHO TỪNG TEST CASE ===
        await ws.send(json.dumps({
            "type": "message",
            "chat_id": chat_id,
            "content": "list files in /tmp using exec tool"
        }))

        start = time.time()
        all_deltas = ""
        all_reasoning = ""
        while time.time() - start < 60:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=10)
                data = json.loads(msg)
                evt = data.get("event", "")
                if evt == "delta":
                    all_deltas += data.get("text", "")
                elif evt == "reasoning_delta":
                    all_reasoning += data.get("text", "")
                elif evt == "turn_end":
                    break
            except asyncio.TimeoutError:
                continue

        # Check for leaks
        for label, text in [("deltas", all_deltas), ("reasoning", all_reasoning)]:
            found = [p for p in LEAKS if p in text]
            if found:
                print(f"⚠️  {label}: LEAKED {found}")
            else:
                print(f"✅ {label}: clean ({len(text)} chars)")
        print(f"Text: {all_deltas[:200]}")

asyncio.run(test())
PYEOF
```

**Message format đúng cho WS:**
```json
{"type": "message", "chat_id": "<from ready event>", "content": "your message here"}
```
**Không dùng** `{"type": "text", "text": "..."}` — sẽ bị ignore.

**WebSocket Event Protocol:**

| Event | Direction | Description |
|-------|-----------|-------------|
| `ready` | Server → Client | Sent on connect. Contains `chat_id`, `client_id` |
| `goal_status` (running) | Server → Client | Agent started processing |
| `reasoning_delta` | Server → Client | Streaming reasoning/thinking tokens |
| `reasoning_end` | Server → Client | Reasoning complete |
| `delta` / `text_delta` | Server → Client | Streaming response text chunks |
| `stream_end` | Server → Client | Text stream complete |
| `turn_end` | Server → Client | Full turn complete. Contains `latency_ms` |
| `goal_status` (idle) | Server → Client | Agent finished |
| Plain text | Client → Server | Send `str` to trigger agent response |
| `{"type":"new_chat"}` | Client → Server | Create new chat session |
| `{"type":"attach","chat_id":"..."}` | Client → Server | Attach to existing chat |

## Bug Investigation Flow

**QUY TẮC:** Khi nhận bug report → **luôn reproduce trước, confirm sau**. Không assume bug tồn tại nếu chưa test thực tế.

### Step 1: Reproduce qua WebSocket

Dùng WebSocket test script để trigger bug trực tiếp, bypass channel (Telegram/Discord):

```bash
# Template: WebSocket chat test — thay <message> theo bug scenario
timeout 120 ssh root@100.68.251.84 'docker exec nanobot-gateway timeout 90 python3 -c "
import asyncio, json, urllib.request
from websockets.asyncio.client import connect

req = urllib.request.Request(\"http://localhost:8765/auth/token\")
req.add_header(\"Authorization\", \"Bearer 0afb95515cc60ef3c290c73b8f0fd6746129ce2a3f2b6460463d83ef31e26b6b\")
token = json.loads(urllib.request.urlopen(req, timeout=5).read())[\"token\"]

async def test():
    async with connect(f\"ws://localhost:8765/?token={token}\") as ws:
        ready = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
        print(f\"ready\", flush=True)

        # === THAY MESSAGE NÀY CHO TỪNG BUG ===
        await ws.send(\"<YOUR TEST MESSAGE HERE>\")
        print(f\"sent\", flush=True)

        full_text = \"\"
        turn_count = 0
        for i in range(150):
            try:
                r = await asyncio.wait_for(ws.recv(), timeout=20)
                d = json.loads(r)
                e = d.get(\"event\")
                if e in (\"delta\", \"text_delta\"):
                    t = d.get(\"text\", \"\")
                    full_text += t
                    print(t, end=\"\", flush=True)
                elif e == \"turn_end\":
                    turn_count += 1
                    print(f\"\\n--- turn_end #{turn_count} ({d.get(\"latency_ms\")}ms) ---\", flush=True)
                    if turn_count >= 3:
                        break
                elif e == \"text\":
                    t = d.get(\"text\", \"\")
                    full_text += t
                    print(f\"\\ntext: {t[:300]}\", flush=True)
                elif e == \"goal_status\":
                    print(f\"goal:{d.get(\"status\")}\", flush=True)
                elif e not in (\"reasoning_delta\", \"reasoning_end\"):
                    print(f\"[{i}] {e}\", flush=True)
            except asyncio.TimeoutError:
                print(\"\\ntimeout\", flush=True)
                break
        print(f\"\\n=== RESULT: {turn_count} turns, {len(full_text)} chars ===\", flush=True)

asyncio.run(test())
"'
```

### Step 2: Check Logs

```bash
# Logs real-time (chạy song song với WebSocket test)
ssh root@100.68.251.84 'docker logs nanobot-gateway -f --tail 50'

# Grep specific patterns
ssh root@100.68.251.84 'docker logs nanobot-gateway --tail 500 | grep -i error'
ssh root@100.68.251.84 'docker logs nanobot-gateway --tail 500 | grep -i "tool_call\|function="'
```

### Step 3: Confirm hoặc Deny

| Kết quả | Action |
|---------|--------|
| **Reproduced** → bug tồn tại | Diagnose root cause → fix → verify fix bằng cùng reproduce script |
| **Cannot reproduce** → bug có thể intermittent | Thử thêm vài lần, thay message, check xem có condition-specific không |
| **Not a bug** → works correctly | Báo lại: "verified working, no issue found" |

### Known Bug Reproduce Scripts

#### Tool Call XML Leak (mimo-v2.5, glm-5.1)

**Symptom:** Channel nhận message chứa `<function=...>` hoặc `<tool_call_none>` thay vì tool execution.
**Root cause:** Model trả tool call dạng XML text trong `content` field thay vì structured `tool_calls` field. Nanobot chỉ parse structured → raw XML leak ra channel.
**Fixed in:** PR #4124 (upstream), deployed in fork.

**Verify fix via WS leak check script above with message:**
```
list files in /tmp using exec tool
```

**Expected (fixed):** ✅ CLEAN — 0 XML patterns in deltas/reasoning, tool executes normally.
**Buggy:** ⚠️ LEAKED — `<function=exec>`, `<parameter=cmd>`, etc. appear in delta events.

**Check provider parser activity in logs:**
```bash
ssh -p 22 root@100.68.251.84 "docker logs nanobot-gateway 2>&1 | grep 'Extracted.*XML tool call'"
# Expected: "Extracted 1 XML tool call(s) from content: ['exec']"
```

**Three-part defense:**
1. Provider parser (`_extract_xml_tool_calls`) — converts XML → structured tool_calls so tools execute
2. Streaming sanitizer (`XmlToolCallSanitizer`) — buffers/strips XML from content deltas
3. Progress hook fix — uses sanitized delta directly (was re-introducing XML from raw buffer)

### Xem Logs Chi Tiết

```bash
# Follow logs real-time
ssh -p 22 root@100.68.251.84 'docker logs nanobot-gateway -f --tail 50'

# Grep errors
ssh -p 22 root@100.68.251.84 'docker logs nanobot-gateway --tail 500 | grep -i error'

# Grep XML/tool call activity
ssh -p 22 root@100.68.251.84 'docker logs nanobot-gateway --tail 500 | grep -i "tool_call\|function=\|Extracted"'

# Check channel status
ssh -p 22 root@100.68.251.84 'docker exec nanobot-gateway nanobot channels status'

# Check full nanobot status (providers, config, channels)
ssh -p 22 root@100.68.251.84 'docker exec nanobot-gateway nanobot status'

# Check container running + image
ssh -p 22 root@100.68.251.84 'docker ps --filter name=nanobot-gateway --format "{{.Image}} | {{.Status}}"'
```

### Interactive Chat Test

```bash
# Chat trực tiếp với nanobot qua CLI (bypass channels)
ssh root@100.68.251.84 'docker exec nanobot-gateway nanobot agent -m "say hello"'
```

### Port Reference

| Port | Service | Usage |
|------|---------|-------|
| `18790` | Health endpoint | GET `/health` → `{"status":"ok"}` |
| `8765` | WebSocket + WebUI | Chat UI, WebSocket multiplex protocol |

**Note:** OpenAI-compatible API (`/v1/chat/completions`, `/v1/models`) chạy bởi `nanobot serve` (command riêng), KHÔNG chạy trong gateway mode. Container hiện chạy `nanobot gateway`.

## Container Runtime

```
nanobot-gateway
├── Image: 100.68.251.84:3005/lucndm/nanobot:latest
├── Command: ["gateway"]
├── User: UID 1000 (nanobot)
├── Ports: 18790 (gateway API/WebUI), 8765 (WebSocket)
├── Volumes:
│   ├── /mnt/user/appdata/nanobot/config → /home/nanobot/.nanobot
│   └── /mnt/user/downloads → /mnt/user/downloads:ro
├── Network: reverse-proxy (external)
├── Limits: 1GB RAM, 1 CPU
└── Restart: always
```

## Path Reference (Critical)

Komodo periphery runs inside a Docker container. Paths differ between **host** and **container**:

| Context | nanobot stack path |
|---------|-------------------|
| **Periphery container** | `/etc/komodo/repos/docker-compose/stacks/llm/nanobot/` |
| **Unraid host** | `/mnt/user/appdata/komodo/komodo-data/repos/docker-compose/stacks/llm/nanobot/` |
| **Dev machine (docker-compose repo)** | `/workspaces/github.com/minhluc-info/docker-compose/stacks/llm/nanobot/` |

**Why the difference:** Periphery container mounts `/mnt/user/appdata/komodo/komodo-data` → `/etc/komodo`. Docker daemon runs on host (via mounted socket). rsync via SSH must use host path. `docker build` can use either (context streamed via socket), but host path is simpler.

## SSH to Unraid

Direct SSH access to unraid via Tailscale VPN:

```bash
# SSH to unraid
ssh root@100.68.251.84

# One-liner commands
ssh root@100.68.251.84 '<command>'

# Examples
ssh root@100.68.251.84 'docker ps --format "{{.Names}}\t{{.Status}}"'
ssh root@100.68.251.84 'docker logs nanobot-gateway --tail 50'
ssh root@100.68.251.84 'ls /mnt/user/appdata/komodo/komodo-data/repos/docker-compose/stacks/llm/nanobot/'
```

### Useful Paths on Unraid Host

| Path | Description |
|------|-------------|
| `/mnt/user/appdata/komodo/komodo-data/repos/` | Komodo git repo clones |
| `/mnt/user/appdata/komodo/komodo-data/repos/docker-compose/` | docker-compose repo clone |
| `/mnt/user/appdata/komodo/komodo-data/repos/docker-compose/stacks/llm/nanobot/` | Nanobot stack dir |
| `/mnt/user/appdata/nanobot/config/` | Nanobot runtime config (mounted into container) |
| `/mnt/user/appdata/komodo/komodo-data/` | Periphery data root (mapped to `/etc/komodo` in container) |

### Non-Interactive: km-exec (GitOps Phase Only)

`km-exec` goes through Komodo API → periphery container. Use **only** khi deploy qua Komodo (Phase 2). Không dùng cho Phase 1 (fast iteration).

```bash
km-exec -s unraid --host '<command>'        # Run on host via Komodo
km-exec nanobot-gateway '<command>'          # Exec into container via Komodo
```

## Deployment Flow

Two-phase approach: **Fast iteration** (build local → scp → deploy) → **GitOps release** (commit+push+Komodo deploy when verified).

### Phase 1: Fast Iteration (Build Local → SCP → Deploy)

Build Docker image from nanobot repo locally, transfer to unraid, restart container. ~3-5 min per iteration.

#### One-Liner: Full Deploy

```bash
# Build → save → scp → load → recreate → verify (all-in-one)
cd /workspaces/github.com/minhluc-info/nanobot && \
docker build -t nanobot:local . && \
docker save nanobot:local | gzip > /tmp/nanobot-local.tar.gz && \
scp /tmp/nanobot-local.tar.gz root@100.68.251.84:/tmp/ && \
ssh -p 22 root@100.68.251.84 "\
  docker load -i /tmp/nanobot-local.tar.gz && \
  docker tag nanobot:local nanobot:latest && \
  docker stop nanobot-gateway && docker rm nanobot-gateway && \
  docker run -d --name nanobot-gateway --restart unless-stopped --network host --user 0:0 \
    -v /mnt/user/appdata/nanobot/config:/home/nanobot/.nanobot \
    -v /mnt/user/appdata/nanobot/workspace:/home/nanobot/.nanobot/workspace \
    nanobot:latest gateway" && \
sleep 5 && ssh -p 22 root@100.68.251.84 "docker logs nanobot-gateway --tail 15"
```

#### Step-by-Step

**Step 1: Build locally**
```bash
cd /workspaces/github.com/minhluc-info/nanobot
docker build -t nanobot:local .
```

**Step 2: Transfer to unraid**
```bash
docker save nanobot:local | gzip > /tmp/nanobot-local.tar.gz
scp /tmp/nanobot-local.tar.gz root@100.68.251.84:/tmp/
```

**Step 3: Load & recreate container**
```bash
ssh -p 22 root@100.68.251.84 "\
  docker load -i /tmp/nanobot-local.tar.gz && \
  docker tag nanobot:local nanobot:latest && \
  docker stop nanobot-gateway && docker rm nanobot-gateway && \
  docker run -d --name nanobot-gateway --restart unless-stopped --network host --user 0:0 \
    -v /mnt/user/appdata/nanobot/config:/home/nanobot/.nanobot \
    -v /mnt/user/appdata/nanobot/workspace:/home/nanobot/.nanobot/workspace \
    nanobot:latest gateway"
```

**Step 4: Verify**
```bash
ssh -p 22 root@100.68.251.84 "docker logs nanobot-gateway --tail 15"
```

#### Notes

- `--network host` simplifies port mapping (uses host network directly)
- `--user 0:0` avoids UID permission issues with config volume
- Config at `/mnt/user/appdata/nanobot/config/` contains `config.json` with all secrets
- Config volume mounts as `/home/nanobot/.nanobot` (default nanobot config path)

### Phase 2: GitOps Release (When Image Verified)

After Phase 1 confirms image works. Commit source changes, push both repos, let Komodo build + deploy.

```bash
# 1. Commit and push nanobot source
cd /workspaces/github.com/minhluc-info/nanobot
git add -A && git commit -m "fix: XML tool call leak for mimo/glm models" && git push

# 2. Sync source to docker-compose repo
rsync -av --delete \
  --exclude='.git' --exclude='.venv' --exclude='__pycache__' \
  --exclude='node_modules' --exclude='*.egg-info' --exclude='dist' --exclude='build' \
  --exclude='.env' --exclude='webui/node_modules' \
  ./ /workspaces/github.com/minhluc-info/docker-compose/stacks/llm/nanobot/nanobot-source/

# 3. Update docker-compose Dockerfile if needed, then commit+push
cd /workspaces/github.com/minhluc-info/docker-compose
git add -A && git commit -m "feat(nanobot): sync source v$(grep '^version' stacks/llm/nanobot/nanobot-source/pyproject.toml | cut -d'"' -f2)" && git push

# 4. Komodo build + deploy
km x run-build nanobot -y && km x deploy-stack nanobot -y
```

### Deploy Flow Summary

```
┌─ Phase 1: Fast Iteration ───────────────────────────────────────────┐
│                                                                      │
│  Fix code → docker build → save/scp/load → restart → verify        │
│       ↑                                                          │   │
│       └──────────────────── fails? retry ────────────────────────┘   │
│                                                                      │
└── passes? ──→ Phase 2: GitOps Release ──────────────────────────────┘
                 │
                 git push nanobot + rsync + git push docker-compose
                 km x run-build nanobot (proper version tags)
                 km x deploy-stack nanobot (GitOps state tracked)
```

## Common Deploy Scenarios

### Config Change Only (config.json, env vars)

No image rebuild needed. Edit config, deploy via Komodo:

Edit `stacks/llm/nanobot/config/config.json` or `komodo/stacks/llm/nanobot.toml`:

```bash
cd /workspaces/github.com/minhluc-info/docker-compose
# TOML changes need sync
git add -A && git commit -m "chore(nanobot): update config" && git push
km x run-sync llm -y && km x deploy-stack nanobot -y
```

Or for fast test: edit config directly on unraid, then `docker compose restart`:
```bash
ssh root@100.68.251.84 'cd /mnt/user/appdata/komodo/komodo-data/repos/docker-compose/stacks/llm/nanobot && \
  vi config/config.json && docker compose restart'
```
When verified, commit to docker-compose repo and Komodo deploy.

### Compose Change Only (ports, limits, volumes)

Edit `stacks/llm/nanobot/compose.yaml`:

```bash
cd /workspaces/github.com/minhluc-info/docker-compose
git add -A && git commit -m "chore(nanobot): update compose" && git push
km x deploy-stack nanobot -y
```

### Change Default Model

Edit env vars in `compose.yaml`:

```yaml
environment:
  - NANOBOT_AGENTS__DEFAULTS__MODEL=mimo-v2.5
```

Or edit `config.json` model presets. Then deploy compose-only.

### Add MCP Server

Edit `config/config.json` under `tools.mcpServers`. The `pre_deploy.command` copies it with secret substitution. TOML sync needed if pre_deploy changed.

## Config & Secrets

### Config Hierarchy (highest priority first)

1. **Environment variables** (compose.yaml `environment`) — `NANOBOT_*__*` convention
2. **config.json** (`/home/nanobot/.nanobot/config.json`) — generated by pre_deploy
3. **Defaults** (code defaults in schema.py)

### Secret Injection Flow

```
Komodo Variables (UI)
  → TOML environment block: [[SECRET_NAME]]
    → Written to .env on periphery during deploy
      → pre_deploy: source .env → sed replace __PLACEHOLDER__ in config.json
```

| Secret | Komodo Variable | Usage |
|--------|----------------|-------|
| `NANOBOT_LITELLM_KEY` | `[[NANOBOT_LITELLM_KEY]]` | LiteLLM virtual key (model access + MCP auth) |
| `NANOBOT_WS_SECRET` | `[[NANOBOT_WS_SECRET]]` | WebSocket auth token |
| `GITHUB_TOKEN` | `[[GITHUB_TOKEN]]` | GitHub API access |

### Config Template Placeholders

`config/config.json` uses `__NAME__` placeholders replaced by sed in pre_deploy:

```bash
sed -i "s|__NANOBOT_LITELLM_KEY__|${NANOBOT_LITELLM_KEY}|g" $CFG
sed -i "s|__NANOBOT_WS_SECRET__|${NANOBOT_WS_SECRET__}|g" $CFG
```

## Dockerfile Source Patches

The Dockerfile applies runtime patches to nanobot after install:

### MCP Timeout Patch

Fixes `list_resources`/`list_prompts` hang (no timeout → agent freezes):

```dockerfile
RUN MCP_FILE=$(python -c "import nanobot.agent.tools.mcp; print(...)") \
    && sed -i 's/await session\.list_resources()/await asyncio.wait_for(session.list_resources(), timeout=15)/' "$MCP_FILE" \
    && sed -i 's/await session\.list_prompts()/await asyncio.wait_for(session.list_prompts(), timeout=15)/' "$MCP_FILE" \
    && sed -i 's/except Exception as e:/except (asyncio.TimeoutError, Exception) as e:/' "$MCP_FILE"
```

**Check if patch still needed:** When upgrading nanobot version, verify if upstream fixed these timeouts. If fixed, remove sed lines from Dockerfile.

## Komodo CLI Reference (Phase 2 Only)

These commands go through Komodo API → periphery. Use **only** for GitOps releases.

```bash
# Connectivity — use Tailscale IP, NOT Cloudflare domain
KOMODO_ADDRESS=http://100.126.172.96:9120

# Build
km x run-build nanobot -y              # Trigger image build
km list builds -n nanobot              # Check build status

# Deploy
km x deploy-stack nanobot -y           # Deploy/redeploy container
km x restart-stack nanobot -y          # Restart without rebuild
km x run-sync llm -y                   # Sync llm category from git

# Monitor
km container -s unraid | grep nanobot  # Container status
km-logs nanobot-gateway -n 100         # Tail logs
km-exec nanobot-gateway nanobot status  # Exec into container
```

## Troubleshooting

### Phase 1 (SSH trực tiếp)

| Issue | Check | Fix |
|-------|-------|-----|
| Build fails | `ssh root@100.68.251.84 'docker logs nanobot-gateway --tail 50'` | Check Dockerfile syntax, rsync completeness |
| Container crash loop | `ssh root@100.68.251.84 'docker logs nanobot-gateway --tail 100'` | Check config.json syntax, secret injection |
| Telegram not responding | `ssh root@100.68.251.84 'docker exec nanobot-gateway nanobot channels status'` | Verify token, check bot conflict |
| LLM errors | `ssh root@100.68.251.84 'docker exec nanobot-gateway nanobot status'` | Model must exist on LiteLLM at `100.68.251.84:4001` |
| Config not updating | `ssh root@100.68.251.84 'cat /mnt/user/appdata/nanobot/config/config.json'` | Verify pre_deploy ran manually |
| Source out of date | `ssh root@100.68.251.84 'ls /mnt/user/appdata/komodo/komodo-data/repos/docker-compose/stacks/llm/nanobot/nanobot-source/pyproject.toml'` | Re-run rsync |
| MCP agent hangs | `ssh root@100.68.251.84 'docker exec nanobot-gateway grep asyncio.wait_for /usr/local/lib/python3.13/site-packages/nanobot/agent/tools/mcp.py'` | Verify Dockerfile timeout patch |

### Phase 2 (GitOps qua Komodo)

| Issue | Check | Fix |
|-------|-------|-----|
| Build fails | `km list builds -n nanobot` | Check Dockerfile syntax, rsync completeness |
| Deploy fails | `km-logs nanobot-gateway -n 100` | Check compose.yaml, TOML config |
| Image pull error | `curl http://100.68.251.84:3005/v2/` | Check Gitea registry accessible |

## Guidelines

- **GitOps mandatory** — never SSH to unraid to run docker commands directly
- **File ownership** — container runs as UID 1000; volumes must `chown 1000:100`
- **Source is gitignored** — nanobot-source is not tracked in docker-compose git; always rsync fresh
- **Pre_deploy is single-line** — Komodo TOML multi-line strings collapse; use `&&` chains only
- **Config template** — secrets go through `[[VAR]]` → `.env` → `sed` → config.json, never hardcoded
- **Patch verification** — after source sync, check if Dockerfile patches are still needed against new version
- **Build before deploy** — source changes require build; config-only changes just need deploy
- **Verify after deploy** — always check logs and container status after deploy completes
