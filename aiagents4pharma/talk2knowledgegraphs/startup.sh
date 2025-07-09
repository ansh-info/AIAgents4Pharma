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

echo "[STARTUP] Milvus is ready. Starting Ollama service..."

# Select correct Ollama image
if [ "$GPU_TYPE" = "amd" ]; then
	OLLAMA_IMAGE="ollama/ollama:rocm"
else
	OLLAMA_IMAGE="ollama/ollama:latest"
fi

# Ensure Docker network exists
docker network inspect milvus >/dev/null 2>&1 || docker network create milvus

echo "[STARTUP] Using image: $OLLAMA_IMAGE"

# Start Ollama container
if [ "$GPU_TYPE" = "nvidia" ]; then
	echo "[STARTUP] Launching Ollama with NVIDIA runtime..."
	docker run -d \
		--name ollama \
		--runtime=nvidia \
		--network milvus \
		-v ollama_data:/root/.ollama \
		-p 11434:11434 \
		-e NVIDIA_VISIBLE_DEVICES=all \
		-e NVIDIA_DRIVER_CAPABILITIES=all \
		--entrypoint /bin/sh \
		"$OLLAMA_IMAGE" \
		-c "ollama serve & sleep 10 && ollama pull nomic-embed-text && tail -f /dev/null"

elif [ "$GPU_TYPE" = "amd" ]; then
	echo "[STARTUP] Launching Ollama with AMD ROCm..."
	docker run -d \
		--name ollama \
		--network milvus \
		-v ollama_data:/root/.ollama \
		-p 11434:11434 \
		--device=/dev/kfd \
		--device=/dev/dri \
		-e ROC_ENABLE_PRE_VEGA=1 \
		-e HSA_ENABLE_SDMA=0 \
		--entrypoint /bin/sh \
		"$OLLAMA_IMAGE" \
		-c "ollama serve & sleep 10 && ollama pull nomic-embed-text && tail -f /dev/null"

else
	echo "[STARTUP] Launching Ollama in CPU mode..."
	OLLAMA_IMAGE=$OLLAMA_IMAGE docker compose up -d ollama
fi

# Wait for Ollama API to be ready
echo "[STARTUP] Waiting for Ollama API..."
until curl -s http://localhost:11434/api/tags >/dev/null 2>&1; do
	echo "[STARTUP] Ollama not ready yet..."
	sleep 5
done

# Pull the model (only for CPU mode; already handled in GPU mode)
if [ "$GPU_TYPE" = "cpu" ]; then
	echo "[STARTUP] Pulling model 'nomic-embed-text' for CPU setup..."
	docker exec ollama ollama pull nomic-embed-text
fi

# Wait for model to appear
echo "[STARTUP] Waiting for model 'nomic-embed-text' to become available..."
until docker exec ollama ollama list | grep -q "nomic-embed-text"; do
	echo "[STARTUP] Model not ready yet..."
	sleep 5
done

echo "[STARTUP] Model is ready. Starting talk2knowledgegraphs application..."

# Configure Docker Compose for talk2knowledgegraphs based on GPU type
if [ "$GPU_TYPE" = "nvidia" ]; then
	echo "[STARTUP] Configuring Docker Compose for NVIDIA GPU..."
	cat > docker-compose-gpu.yml << 'EOF'
services:
  talk2knowledgegraphs:
    platform: linux/amd64
    image: virtualpatientengine/talk2knowledgegraphs:latest
    container_name: talk2knowledgegraphs
    ports:
      - "8501:8501"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: ["gpu"]
              device_ids: ["0"]
    env_file:
      - .env
    restart: unless-stopped
    networks:
      - milvus

networks:
  milvus:
    external: true
    name: milvus
EOF
	COMPOSE_FILE="docker-compose-gpu.yml"
else
	echo "[STARTUP] Using CPU-only configuration..."
	COMPOSE_FILE="docker-compose.yml"
fi

# Start the main application with appropriate configuration
docker compose -f $COMPOSE_FILE up -d talk2knowledgegraphs

# Wait a moment for the application to start
sleep 10

# Clean up temporary files
echo "[STARTUP] Cleaning up temporary files..."
rm -f milvus-docker-compose.yml
if [ "$GPU_TYPE" = "nvidia" ]; then
	rm -f docker-compose-gpu.yml
	echo "[STARTUP] Removed docker-compose-gpu.yml"
fi
echo "[STARTUP] Removed milvus-docker-compose.yml"

echo "[STARTUP] System fully running at: http://localhost:8501"
echo "[STARTUP] Milvus API available at: http://localhost:19530"
echo "[STARTUP] Ollama API available at: http://localhost:11434"