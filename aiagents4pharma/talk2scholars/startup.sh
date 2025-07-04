#!/bin/bash
set -e

FORCE_CPU=false

# Parse arguments
while [[ $# -gt 0 ]]; do
	case "$1" in
	--cpu)
		FORCE_CPU=true
		shift
		;;
	*)
		shift
		;;
	esac
done

echo "[STARTUP] Detecting hardware configuration..."

GPU_TYPE="cpu"
ARCH=$(uname -m)

if [ "$FORCE_CPU" = true ]; then
	echo "[STARTUP] --cpu flag detected. Forcing CPU mode."
	GPU_TYPE="cpu"

elif command -v nvidia-smi >/dev/null 2>&1; then
	echo "[STARTUP] Hardware configuration: NVIDIA GPU detected."
	GPU_TYPE="nvidia"

elif command -v lspci >/dev/null 2>&1 && lspci | grep -i amd | grep -iq vga; then
	echo "[STARTUP] Hardware configuration: AMD GPU detected."
	GPU_TYPE="amd"

elif [[ "$ARCH" == "arm64" || "$ARCH" == "aarch64" ]]; then
	echo "[STARTUP] Hardware configuration: Apple Silicon (arm64) detected. Metal acceleration is not available inside Docker. Running in CPU mode."
	GPU_TYPE="cpu"

else
	echo "[STARTUP] Hardware configuration: No supported GPU detected. Running in CPU mode."
	GPU_TYPE="cpu"
fi

# Download appropriate Milvus docker-compose file
if [ "$GPU_TYPE" = "cpu" ]; then
	echo "[STARTUP] Downloading Milvus CPU docker-compose file..."
	wget https://github.com/milvus-io/milvus/releases/download/v2.6.0-rc1/milvus-standalone-docker-compose.yml -O milvus-docker-compose.yml
else
	echo "[STARTUP] Downloading Milvus GPU docker-compose file..."
	wget https://github.com/milvus-io/milvus/releases/download/v2.6.0-rc1/milvus-standalone-docker-compose-gpu.yml -O milvus-docker-compose.yml
fi

# Ensure Docker networks exist
docker network inspect app-network >/dev/null 2>&1 || docker network create app-network

echo "[STARTUP] Starting Milvus with $GPU_TYPE configuration..."

# Start Milvus services
docker compose -f milvus-docker-compose.yml up -d

# Wait for Milvus API to be ready
echo "[STARTUP] Waiting for Milvus API..."
MILVUS_READY=false
MAX_ATTEMPTS=30
ATTEMPT=0

while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
	if curl -s http://localhost:19530/health >/dev/null 2>&1; then
		echo "[STARTUP] Milvus health check passed"
		MILVUS_READY=true
		break
	else
		echo "[STARTUP] Milvus not ready yet... (attempt $((ATTEMPT + 1))/$MAX_ATTEMPTS)"
		sleep 10
		ATTEMPT=$((ATTEMPT + 1))
	fi
done

if [ "$MILVUS_READY" = false ]; then
	echo "[STARTUP] ERROR: Milvus failed to start after $MAX_ATTEMPTS attempts"
	echo "[STARTUP] Checking Milvus logs..."
	docker logs milvus-standalone --tail 20
	echo "[STARTUP] Checking if port 19530 is in use..."
	netstat -ln | grep 19530 || lsof -i :19530 || echo "Port 19530 not found"
	exit 1
fi

# Additional verification - try to connect to Milvus API
echo "[STARTUP] Testing Milvus API endpoints..."
curl -s http://localhost:19530/health && echo " - Health endpoint OK" || echo " - Health endpoint FAILED"

# Try to list collections (this tests if Milvus is actually functional)
if command -v python3 >/dev/null 2>&1; then
	python3 -c "
import sys
try:
    from pymilvus import connections, Collection
    connections.connect('default', host='127.0.0.1', port='19530')
    print('SUCCESS: Python Milvus connection test passed')
    connections.disconnect('default')
except Exception as e:
    print(f'FAILED: Python Milvus connection test failed: {e}')
    sys.exit(1)
" || echo "[STARTUP] Python Milvus test failed - this may be normal if pymilvus is not installed on host"
fi

echo "[STARTUP] Milvus is ready. Connecting talk2scholars to Milvus network..."

# Connect the talk2scholars container to the milvus network after it's created
MILVUS_NETWORK=$(docker network ls --format "table {{.Name}}" | grep milvus | head -1)
if [ -n "$MILVUS_NETWORK" ]; then
	echo "[STARTUP] Found Milvus network: $MILVUS_NETWORK"
else
	echo "[STARTUP] Warning: Could not find Milvus network, using default bridge"
	MILVUS_NETWORK="bridge"
fi

echo "[STARTUP] Starting talk2scholars application..."

# Check if .env file exists and create a temporary compose file accordingly
if [ -f ".env" ]; then
	echo "[STARTUP] Found .env file, using it for environment variables..."
	# Create a temporary docker-compose file with env_file
	cat >docker-compose-temp.yml <<'EOF'
services:
  talk2scholars:
    platform: linux/amd64
    image: virtualpatientengine/talk2scholars:latest
    container_name: talk2scholars
    ports:
      - "8501:8501"
    env_file:
      - .env
    restart: unless-stopped
    networks:
      - app-network

networks:
  app-network:
    external: true
    name: app-network
EOF
	docker compose -f docker-compose-temp.yml up -d talk2scholars
	rm docker-compose-temp.yml
else
	echo "[STARTUP] No .env file found, starting without environment file..."
	docker compose up -d talk2scholars
fi

# Connect talk2scholars to the milvus network
if [ "$MILVUS_NETWORK" != "bridge" ]; then
	echo "[STARTUP] Connecting talk2scholars to Milvus network..."
	docker network connect "$MILVUS_NETWORK" talk2scholars || echo "[STARTUP] Warning: Could not connect to Milvus network"
fi

echo "[STARTUP] System fully running at: http://localhost:8501"
echo "[STARTUP] Milvus API available at: http://localhost:19530"
