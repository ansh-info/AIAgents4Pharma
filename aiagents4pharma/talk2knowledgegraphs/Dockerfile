# Dockerfile for the talk2knowledgegraphs application
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Install system dependencies required for compiling Python packages
RUN apt-get update && apt-get install -y \
  g++ \
  build-essential \
  cmake \
  curl \
  libopenblas-dev \
  libomp-dev \
  ninja-build \
  && rm -rf /var/lib/apt/lists/*

# Install RAPIDS + cudf
RUN conda install -y -n base -c rapidsai -c conda-forge -c nvidia \
  python=3.12 cudf=25.06 "cuda-version>=12.0,<=12.8"

# Install pip packages
COPY requirements.txt /app
RUN pip install --upgrade pip wheel && \
  pip install --retries 15 --timeout 180 --no-cache-dir -r /app/requirements.txt && \
  conda clean -afy && rm -rf ~/.cache/pip

# Copy app code
COPY aiagents4pharma/talk2knowledgegraphs /app/aiagents4pharma/talk2knowledgegraphs
COPY docs /app/docs
COPY app /app/app

# # Copy the data loader script to the correct location
# COPY milvus_data_dump.py /app/aiagents4pharma/talk2knowledgegraphs/

# Create cache directory structure
RUN mkdir -p /app/aiagents4pharma/talk2knowledgegraphs/tests/files

# Set Python path to include the app directory
ENV PYTHONPATH="/app"

# Copy and set up the entrypoint script
COPY aiagents4pharma/talk2knowledgegraphs/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Expose the default Streamlit port
EXPOSE 8501

# Set the entrypoint
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Default command (can be overridden)
CMD ["streamlit", "run", "/app/app/frontend/streamlit_app_talk2knowledgegraphs.py", "--server.port=8501", "--server.address=0.0.0.0"]