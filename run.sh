#!/usr/bin/env bash
set -e

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info() { echo -e "${GREEN}[ShaktiDB]${NC} $*"; }
warn() { echo -e "${YELLOW}[ShaktiDB]${NC} $*"; }

# ── Auto-generate API key if .env is missing or key is unset ─────────────────
ENV_FILE="./backend/.env"
if [[ ! -f "$ENV_FILE" ]]; then
    cp ./backend/.env.example "$ENV_FILE" 2>/dev/null || touch "$ENV_FILE"
fi

if ! grep -qE "^SHAKTIDB_API_KEY=.+" "$ENV_FILE" 2>/dev/null; then
    KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))" 2>/dev/null || \
          node -e "console.log(require('crypto').randomBytes(32).toString('hex'))")
    echo "SHAKTIDB_API_KEY=${KEY}" >> "$ENV_FILE"
    info "Generated API key and saved to $ENV_FILE"
fi

# ── Detect GPU ────────────────────────────────────────────────────────────────
USE_GPU=false
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    # Also check NVIDIA Container Toolkit is available to Docker
    if docker info 2>/dev/null | grep -q "nvidia"; then
        USE_GPU=true
    else
        warn "NVIDIA GPU found but NVIDIA Container Toolkit not configured for Docker."
        warn "Running on CPU. To enable GPU: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    fi
fi

# ── Launch ────────────────────────────────────────────────────────────────────
if [[ "$USE_GPU" == "true" ]]; then
    info "GPU detected — launching with CUDA support."
    COMPOSE_CMD="docker-compose -f docker-compose.yml -f docker-compose.gpu.yml"
else
    info "Launching in CPU mode."
    COMPOSE_CMD="docker-compose -f docker-compose.yml"
fi

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  ShaktiDB AI Pipeline${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "  Frontend : ${CYAN}http://localhost:12001${NC}"
echo -e "  Backend  : ${CYAN}http://localhost:12000${NC}"
echo -e "  Mode     : ${CYAN}$([ "$USE_GPU" == "true" ] && echo "GPU (CUDA)" || echo "CPU")${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

$COMPOSE_CMD up --build "$@"
