#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"
VENV="$SCRIPT_DIR/.venv"
LOG_DIR="$SCRIPT_DIR/logs"
ENV_FILE="$BACKEND_DIR/.env"
ENV_EXAMPLE="$BACKEND_DIR/.env.example"

BACKEND_PORT=12000
FRONTEND_PORT=12001

mkdir -p "$LOG_DIR"

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${GREEN}[start.sh]${NC} $*"; }
warn()    { echo -e "${YELLOW}[start.sh]${NC} $*"; }
error()   { echo -e "${RED}[start.sh]${NC} $*"; }

# ── Bootstrap .env ────────────────────────────────────────────────────────────
if [[ ! -f "$ENV_FILE" ]]; then
    if [[ -f "$ENV_EXAMPLE" ]]; then
        cp "$ENV_EXAMPLE" "$ENV_FILE"
        info "Created $ENV_FILE from .env.example"
    else
        touch "$ENV_FILE"
        info "Created empty $ENV_FILE"
    fi
fi

# Helper: read a value from .env (returns empty string if not set / commented out)
_env_get() {
    local key="$1"
    grep -E "^${key}=" "$ENV_FILE" 2>/dev/null | tail -1 | cut -d'=' -f2- | tr -d '"' | tr -d "'" || true
}

# Helper: set or update a key in .env
_env_set() {
    local key="$1" val="$2"
    if grep -qE "^${key}=" "$ENV_FILE" 2>/dev/null; then
        sed -i "s|^${key}=.*|${key}=${val}|" "$ENV_FILE"
    else
        echo "${key}=${val}" >> "$ENV_FILE"
    fi
}

# ── Auto-generate SHAKTIDB_API_KEY if missing ─────────────────────────────────
EXISTING_KEY=$(_env_get "SHAKTIDB_API_KEY")
if [[ -z "$EXISTING_KEY" || "$EXISTING_KEY" == "change_me_to_a_random_64_char_hex_string" ]]; then
    GENERATED_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
    _env_set "SHAKTIDB_API_KEY" "$GENERATED_KEY"
    info "Generated new SHAKTIDB_API_KEY and saved to .env"
    EXISTING_KEY="$GENERATED_KEY"
fi

# ── Set sensible defaults for non-secret config if missing ────────────────────
[[ -z "$(_env_get CORS_ORIGINS)" ]]    && _env_set "CORS_ORIGINS"   "http://localhost:${FRONTEND_PORT},http://localhost:5173,http://localhost:3000"
[[ -z "$(_env_get MODEL_NAME)" ]]      && _env_set "MODEL_NAME"     "BAAI/bge-m3"
[[ -z "$(_env_get EMBEDDING_DIM)" ]]   && _env_set "EMBEDDING_DIM"  "1024"
[[ -z "$(_env_get HOST)" ]]            && _env_set "HOST"           "0.0.0.0"
[[ -z "$(_env_get PORT)" ]]            && _env_set "PORT"           "$BACKEND_PORT"
[[ -z "$(_env_get RELOAD)" ]]          && _env_set "RELOAD"         "false"

# ── Cleanup on exit ───────────────────────────────────────────────────────────
PIDS=()
cleanup() {
    echo ""
    info "Shutting down..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null
    info "Done."
}
trap cleanup EXIT INT TERM

# ── Backend ───────────────────────────────────────────────────────────────────
info "Starting backend on port ${BACKEND_PORT}..."

if [[ ! -f "$VENV/bin/uvicorn" ]]; then
    error "Virtual environment not found at $VENV"
    error "Run:  python3 -m venv $VENV && $VENV/bin/pip install -r $BACKEND_DIR/requirements.txt"
    exit 1
fi

cd "$BACKEND_DIR"
"$VENV/bin/uvicorn" app.main:app \
    --host 0.0.0.0 \
    --port "$BACKEND_PORT" \
    > "$LOG_DIR/backend.log" 2>&1 &
BACKEND_PID=$!
PIDS+=($BACKEND_PID)

# Wait for backend to be ready (up to 20s)
info "Waiting for backend..."
for i in $(seq 1 20); do
    if curl -s "http://localhost:${BACKEND_PORT}/health" > /dev/null 2>&1; then
        info "Backend ready (PID $BACKEND_PID)"
        break
    fi
    if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
        error "Backend process died. Check $LOG_DIR/backend.log"
        tail -20 "$LOG_DIR/backend.log"
        exit 1
    fi
    sleep 1
done

# ── Frontend ──────────────────────────────────────────────────────────────────
info "Starting frontend on port ${FRONTEND_PORT}..."

if [[ ! -d "$FRONTEND_DIR/node_modules" ]]; then
    warn "node_modules not found — running npm install..."
    cd "$FRONTEND_DIR"
    npm install
fi

cd "$FRONTEND_DIR"
npm run dev -- --host 0.0.0.0 --port "$FRONTEND_PORT" \
    > "$LOG_DIR/frontend.log" 2>&1 &
FRONTEND_PID=$!
PIDS+=($FRONTEND_PID)

# Wait for frontend to be ready (up to 20s)
info "Waiting for frontend..."
for i in $(seq 1 20); do
    if curl -s "http://localhost:${FRONTEND_PORT}" > /dev/null 2>&1; then
        info "Frontend ready (PID $FRONTEND_PID)"
        break
    fi
    if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
        error "Frontend process died. Check $LOG_DIR/frontend.log"
        tail -20 "$LOG_DIR/frontend.log"
        exit 1
    fi
    sleep 1
done

# ── Ready ─────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  ShaktiDB AI Pipeline is running${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "  Frontend : ${YELLOW}http://localhost:${FRONTEND_PORT}${NC}"
echo -e "  Backend  : ${YELLOW}http://localhost:${BACKEND_PORT}${NC}"
echo -e "  Logs     : ${YELLOW}$LOG_DIR/${NC}"
echo ""
echo -e "  ${CYAN}API Key  : ${EXISTING_KEY}${NC}"
echo -e "  ${CYAN}(saved to $ENV_FILE)${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
info "Press Ctrl+C to stop."

# Keep running until interrupted
wait
