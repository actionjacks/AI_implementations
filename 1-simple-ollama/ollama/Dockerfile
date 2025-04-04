FROM nvidia/cuda:11.2.0-base-ubuntu22.04 AS base

# Use a base image with CUDA support if using GPU, otherwise use Ubuntu
FROM ubuntu:22.04

# Set non-interactive mode to avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install required dependencies
RUN apt update && apt install -y \
    curl \
    git \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Expose Ollama's default port (if needed)
EXPOSE 11436

WORKDIR /app

COPY ./.env /app/.env

# https://github.com/ollama/ollama/blob/main/README.md
# Copy context files (custom-model data) Based on this file it can respond based on this data. (Customize a prompt)
COPY ./context/doc.txt /app/context/doc.txt

# Copy entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Using ENTRYPOINT to make the container active and responsible for running Ollam.
ENTRYPOINT ["/entrypoint.sh"]

