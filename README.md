# evaluation-app

Streamlit app for evaluating chatbot transcripts with readability metrics and optional LLM-based extraction via OpenRouter.

## Run

```powershell
python -m streamlit run streamlit_app.py
```

## What it does

- Upload a JSON transcript containing one or more chatbot conversations.
- Validates message structure (`role` in `user|assistant|system` and non-empty `content`).
- Computes readability metrics on assistant messages:
  - Automated Readability Index (ARI)
  - Flesch Reading Ease
  - Dale–Chall score (uses `dale_chall_easy_words.txt`)
- Displays pass/fail thresholds and summary pass rates; copy tables as CSV.
- Optional “Accuracy Performance” section uses OpenRouter to:
  - Summarize user context from user messages.
  - Extract product recommendations from assistant messages.

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

## OpenRouter setup (optional)

To run the “Accuracy Performance” section, add your OpenRouter API key in the sidebar
and select an evaluation model. If no key is provided, readability metrics still work.
