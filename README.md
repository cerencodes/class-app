# evaluation-app

Streamlit multipage app for evaluating chatbot transcripts and running structured multi-model ratings on CSV datasets with OpenRouter-backed models.

## Run

```powershell
python -m streamlit run streamlit_app.py
```

## Pages

- `Chatbot Evaluator`: Upload transcript JSON, validate structure, compute readability metrics, and optionally run OpenRouter-based extraction.
- `Customerbot`: Configure a rating prompt, score CSV rows with two models in parallel, compare agreement, and run a judge model on disagreement rows.

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

`Customerbot` is now a structured rating workflow rather than a simple side-by-side generation demo.

### Step 1: Select Models and Upload Data

- Enter an OpenRouter API key.
- Select two different models to act as raters.
- Upload a CSV with `context` and `content` columns. `row_id` is optional.
- Set temperature for Step 3 generation.

### Step 2: Define the Rating Prompt

- Configure the prompt in sections:
  - `Role`
  - `Task`
  - `Construct Definition`
  - `Inclusion Criteria`
  - `Exclusion Criteria`
  - `Constraints`
  - `Examples`
- The default prompt is empathy-focused, but the workflow is now generic and can be reused for other constructs.

### Step 3: Generate Structured Ratings

- The app sends each uploaded row to both selected models in parallel.
- The system prompt requires valid JSON output as a flat object.
- For scored dimensions, the expected naming convention is:
  - `<dimension>_score`
  - `<dimension>_reasoning`
- Scores are expected to be integer Likert values from 1 to 5.
- Optional aggregate fields are allowed, for example `average_empathy_score`.
- The app dynamically preserves arbitrary returned fields instead of hard-coding empathy columns.
- Results are exported with normalized column names in the format:
  - `<model_prefix>__<field>`
- The app shows:
  - per-model run summaries
  - the scored dataset
  - a preview table
  - Krippendorff's alpha across shared scored dimensions
- Download options:
  - `customerbot-results.json`
  - `customerbot-results.csv`

### Step 4: Optimize Reasoning

- Upload a Step 3 CSV output file.
- The app detects disagreement rows where at least two model ratings differ.
- A judge model reviews disagreement cases and returns JSON with:
  - `disagreement_summary`
  - `sources_of_disagreement`
  - `recommended_edits`
- Step 4 accepts both:
  - the current dynamic column format: `<model_prefix>__<field>`
  - older legacy files that used `<model>_score` / `<model>_reasoning` / `<model>_error`
- Judge output can be downloaded as `customerbot-step-4-judge.json`.

## Customerbot CSV Requirements

Input CSV for Step 3 must contain:

- `context`
- `content`

Optional:

- `row_id`

## Customerbot Output Shape

Example model response shape expected by Step 3:

```json
{
  "individual_attention_score": 1,
  "individual_attention_reasoning": "Reason here.",
  "caring_tone_score": 1,
  "caring_tone_reasoning": "Reason here.",
  "customer_first_orientation_score": 1,
  "customer_first_orientation_reasoning": "Reason here.",
  "need_understanding_score": 1,
  "need_understanding_reasoning": "Reason here.",
  "average_empathy_score": 1.0
}
```

This schema is not fixed to empathy. Any construct can use different dimension names as long as scored dimensions follow the `<dimension>_score` / `<dimension>_reasoning` convention.

## Transcript Format

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

## OpenRouter Setup

Enter your OpenRouter API key on the relevant page before running any model-based analysis.

- `Chatbot Evaluator` can still run validation and readability analysis without an API key.
- `Customerbot` Step 3 and Step 4 require an API key.
