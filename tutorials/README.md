# models that need to be downloaded to ollama:

```bash
ollama pull mxbai-embed-large   
ollama pull llama3:latest               
```

# python - notes
```bash
python -m venv venv
./venv/Scripts/activate
pip install -r ./requirements.txt
```

login huggingface
```bash
huggingface-cli login 
```

enable CUDA -https://download.pytorch.org/whl/cu128 (128 - version)
```baseh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```