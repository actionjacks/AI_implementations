# 1. Installing NVIDIA Container Toolkit

First, ensure that NVIDIA drivers are installed on the host machine. Then, install the NVIDIA Container Toolkit:

```bash
# Add the NVIDIA Container Toolkit repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Update packages and install the NVIDIA Container Toolkit
sudo apt-get update
sudo apt-get install -y nvidia-docker2

# Restart Docker
sudo systemctl restart docker
```

# Limitations
When starting the containers, the models declared for download require time to be fetched.
Allow sufficient time for the models to be downloaded.
Additionally, the available memory allocated for running models in Docker may also be a limiting factor.

OpenWebUI:
```json
http://localhost:8333
```

Ollama API: 
```json
http://localhost:11436
```

Go:
```json
http://localhost:8081 # note that the goapp application in .env listens on (8080).
```

# Notes (TODO - remove) In order not to copy all the time.
```bash
docker exec -it ollama-ollamadeepseek-1 sh
```

```bash
docker logs ollama-ollamadeepseek-1
```

```bash
ollama run llama3:8b "Hello, how are you?"
```