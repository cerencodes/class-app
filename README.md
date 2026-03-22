# evaluation-app

Streamlit multipage app for evaluating chatbot transcripts and generating synthetic customer data with OpenRouter-backed models.

## Run

```powershell
python -m streamlit run streamlit_app.py
```

## Pages

- `Chatbot Evaluator`: Upload transcripts, validate them, compute readability metrics, and run optional LLM-based extraction.
- `Customerbot`: Send one prompt to three language models in parallel and compare synthetic customer data outputs side by side.

## Chatbot Evaluator

- Upload a JSON transcript containing one or more chatbot conversations.
- Validate message structure (`role` in `user|assistant|system` and non-empty `content`).
- Compute readability metrics on assistant messages:
  - Automated Readability Index (ARI)
  - Flesch Reading Ease
  - Flesch-Kincaid Grade Level
  - Gunning Fog Index
  - SMOG Index
  - Dale-Chall score
- Display pass/fail thresholds and summary pass rates.
- Run optional OpenRouter-based extraction to summarize user context and identify products recommended by the assistant.

## Customerbot

- Select three models.
- Provide one shared system prompt and one shared user prompt.
- Run all three model calls concurrently through OpenRouter.
- Review timing and output side by side.
- Download the combined results as JSON.

## Transcript format

Upload a JSON object or list of objects shaped like:

```json
[
  {
    "conversation_id": "conversation_1",
    "model": "gpt-4o-mini",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "I need a lightweight tent for backpacking."},
      {"role": "assistant", "content": "Consider the Big Agnes Copper Spur."}
    ]
  }
]
```

`conversation_id` and `model` are optional; missing values are filled in.

## OpenRouter setup

Enter your OpenRouter API key in the relevant page before running any model-based analysis.
If no key is provided, the transcript validation and readability features still work.
