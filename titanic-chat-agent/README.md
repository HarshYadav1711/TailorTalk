# Titanic Chat Agent

## Project Overview

This is a local-first chatbot for the Titanic dataset.

It takes natural language questions, routes them through deterministic pandas tools, and returns:
- a concise text answer
- the tool that was used
- an optional real matplotlib chart

No paid APIs are used. The LLM runs locally with Ollama.

## Architecture

The project is split into a small backend and frontend:

- `backend/main.py`: FastAPI app with one `POST /chat` endpoint
- `backend/agent.py`: LangChain agent setup with Ollama (`temperature=0`)
- `backend/tools.py`: deterministic pandas tools (calculations + visualization)
- `backend/data_loader.py`: Titanic dataset loading and in-memory caching
- `backend/evaluation.py`: fixed-query evaluation script with accuracy output
- `frontend/app.py`: Streamlit chat interface

## Why Tool-Based Design

The agent is only responsible for choosing the right tool and formatting a final answer.

All math and data operations happen in Python functions (pandas + matplotlib). This keeps results deterministic and avoids model-side arithmetic errors.

## Hallucination Prevention

The system prompt enforces strict behavior:
- always call tools for computations
- never guess
- return `"I cannot compute that"` for unsupported requests

Tools validate inputs (column names, numeric constraints, empty data checks), so invalid operations fail clearly instead of producing fabricated answers.

## Setup

1. Install Python dependencies:

```bash
pip install -r backend/requirements.txt
```

2. Install Ollama from the official site and start it.

3. Pull one supported local model:

```bash
ollama pull mistral
```

You can also use:

```bash
ollama pull llama3
```

If you switch model, update `model_name` in `backend/agent.py`.

## Run Backend

From `titanic-chat-agent/`:

```bash
uvicorn backend.main:app --reload
```

Backend runs at `http://localhost:8000`.

## Run Frontend

From `titanic-chat-agent/`:

```bash
streamlit run frontend/app.py
```

## Run Evaluation

From `titanic-chat-agent/`:

```bash
python backend/evaluation.py
```

The script runs 10 fixed test queries, compares outcomes against expected tool/keyword checks, and prints overall accuracy.

