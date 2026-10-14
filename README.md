# Personalized Study Assistant Chatbot

A study-question answering prototype with a Flask chat page, local study notes, and a retrieval pipeline.

## What is included

- A browser chat page and `POST /ask` endpoint.
- A script for fine-tuning `deepset/roberta-base-squad2` on a small SQuAD v2 subset.
- A script for indexing `.txt` files from `study_materials/` in FAISS.
- LangChain retrieval code that attempts to return an answer and source excerpts.

## Status

Prototype with setup and integration gaps. The repository does not contain a complete trained model or FAISS index, and it has no dependency manifest or automated test suite. It is not ready to run unchanged.

Two source issues need resolving first:

1. `vectorize.py` embeds documents with `distilbert-base-uncased`, while `rag.py` queries with `sentence-transformers/all-MiniLM-L6-v2`. Use the same embedding model for both and rebuild the index.
2. `rag.py` wraps a Transformers extractive question-answering pipeline as a LangChain LLM. That interface needs checking or adapting before answers can be relied on.

## Setup workflow

The source uses Python, Flask, PyTorch, Transformers, Hugging Face Datasets, tqdm, LangChain, sentence-transformers, FAISS, and a text document loader. No compatible package versions are recorded, so a reproducible installation command is not available yet. Model and dataset downloads require internet access.

After preparing a compatible environment and resolving the issues above, the source entry points are:

```bash
python fine_tune.py
python vectorize.py
python app.py
```

Run them from the repository root. Training writes to `fine_tuned_model/`; vectorization writes to `faiss_index/`; the Flask page uses port `5000`. The current server starts in debug mode and binds to all interfaces, so change those settings before sharing it beyond local development.

A local workflow is to add study notes to `study_materials/`, rebuild the index, open `http://localhost:5000`, and ask a question covered by those notes. This workflow has not been verified end to end.

## Source guide

- `app.py` — HTML chat interface and Flask routes.
- `fine_tune.py` — question-answering model training.
- `vectorize.py` — study-note loading and FAISS indexing.
- `rag.py` — model/index loading and retrieval chain.
- `study_materials/` — sample history and science notes.

## Checks and limitations

No automated checks are defined. `fine_tuned_model/` contains only partial tokenizer files. Retrieval initialization can fail while the web page still starts. The FAISS loader enables pickle-based deserialization; only load an index you created and trust. Answer accuracy and useful study recommendations have not been established.

## License

No license file is included.
