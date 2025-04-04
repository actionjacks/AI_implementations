2-rag: - based on https://github.com/rag-web-ui/rag-web-ui  (I changed the default implementation to be based on the ollama language model)
    After the containers are launched it is necessary to download the ollama models manually.
    ```
    docker exec -it 2-rag-ollama-1 sh
    ```
    ```
    ollama pull deepseek-r1:7b
    ollama pull omic-embed-text
    ```