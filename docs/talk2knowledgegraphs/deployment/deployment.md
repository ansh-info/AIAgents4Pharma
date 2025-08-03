
# 🛠️ Deployment Guide for Talk2KnowledgeGraphs (T2KG)

This step-by-step tutorial helps you deploy **Talk2KnowledgeGraphs (T2KG)** on your local machine.

> **Note:** This deployment guide assumes that you have access to a machine with **NVIDIA GPU(s)**.

---

## ✅ Step 1: Install Conda

Install the Anaconda Python distribution, which simplifies package and environment management.

```bash
wget https://repo.anaconda.com/archive/Anaconda3-2025.06-0-Linux-x86_64.sh
bash Anaconda3-2025.06-0-Linux-x86_64.sh
source ~/.bashrc
```

---

## ✅ Step 2: Install NVIDIA CUDA Toolkit

Install NVIDIA CUDA libraries to enable GPU-accelerated computation required for model inference.

```bash
sudo apt update
sudo apt install nvidia-cuda-toolkit
```

---

## ✅ Step 3: Install NVIDIA Container Toolkit for Docker

This allows Docker containers to access your GPU using the NVIDIA runtime.

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

```bash
sudo apt-get update
```

```bash
export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.17.8-1
sudo apt-get install -y \
    nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}
```

> For more details, see the [official NVIDIA documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.17.8/install-guide.html).

---

## ✅ Step 4: Restart Docker

Reload Docker to apply the NVIDIA runtime settings.

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```

---

## ✅ Step 5: Install Python 3.12 Virtual Environment

This is optional but recommended if you're running code outside Docker and want isolated Python environments.

```bash
sudo apt install python3.12-venv
```

---

## ✅ Step 6: Clone the AIAgents4Pharma Repository

Download the T2KG codebase which includes Docker configs, notebooks, and the Streamlit frontend.

```bash
mkdir repositories
cd repositories
git clone https://github.com/VirtualPatientEngine/AIAgents4Pharma
cd AIAgents4Pharma
```

---

## ✅ Step 7: Configure the `.env` File

Copy the example environment file and update paths, keys, and credentials as needed.

```bash
cd aiagents4pharma/talk2knowledgegraphs
cp .env.example .env
```

> ✏️ Make sure to fill in all fields, especially the absolute `DATA_DIR` path.

---

## ✅ Step 8: Launch Dockerized T2KG Pipeline

This starts the backend (Milvus, API server) and frontend (Streamlit UI) in containers.

```bash
docker compose up -d
```