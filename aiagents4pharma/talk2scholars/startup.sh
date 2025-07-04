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

echo "[STARTUP] Milvus is ready. Starting talk2scholars application..."

# Start the main application (it will automatically connect to milvus network)
docker compose up -d talk2scholars

# Wait a moment for the application to start
sleep 10

# Test connectivity
echo "[STARTUP] Testing Milvus connectivity..."
docker exec talk2scholars sh -c "
    echo 'Testing connection to milvus-standalone:19530...'
    curl -s http://milvus-standalone:19530/health && echo 'SUCCESS: Milvus is accessible' || echo 'FAILED: Cannot reach Milvus'
"

# Clean up the downloaded milvus docker-compose file
echo "[STARTUP] Cleaning up temporary files..."
rm -f milvus-docker-compose.yml
echo "[STARTUP] Removed milvus-docker-compose.yml"

echo "[STARTUP] System fully running at: http://localhost:8501"
echo "[STARTUP] Milvus API available at: http://localhost:19530"
