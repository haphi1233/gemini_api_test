# Gemini Embeddings (Python) Quick Test

This is a minimal project to test Google Gemini Embeddings using Python.

## Requirements
- Python 3.9+
- A valid Google API Key with access to Gemini models

## Setup
1. Create a virtual environment (optional but recommended)
   - Windows (PowerShell)
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - macOS/Linux
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
2. Install dependencies
   ```bash
   python -m pip install -r requirements.txt
   ```
3. Provide your API key
   - Copy `.env.example` to `.env` and set `GOOGLE_API_KEY`.
   - Alternatively, pass `--api-key` argument when running the script (not recommended for shared machines since it may be stored in shell history).

## Models
- Default: `text-embedding-004`
- You can also try: `text-embedding-004-multilingual` (if enabled for your account)

## Usage
Single text:
```bash
python -m src.embedding_test --text "Xin chao Gemini"
```

Multiple texts (arguments):
```bash
python -m src.embedding_test --text "hello" "xin chao" "bonjour"
```

From a text file (one line per text):
```bash
python -m src.embedding_test --file ./samples.txt
```

Specify model and task type:
```bash
python -m src.embedding_test --text "what is rAG?" --model text-embedding-004 --task retrieval_query
```

Override API key via CLI:
```bash
python -m src.embedding_test --text "xin chao" --api-key "YOUR_API_KEY"
```

## Output
- Prints the embedding dimension and the first few values for each text.
- Use `--save ./embeddings.json` to store full vectors.

## Notes
- For retrieval use-cases, set `--task retrieval_document` for documents and `--task retrieval_query` for queries to improve performance.
- This script uses `google-generativeai` (server API) not Vertex AI SDK.
