import json
import math
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openrouter_utils import timed_openrouter_chat_completion


DEFAULT_MODELS = [
    "openai/gpt-4o-mini",
    "google/gemini-2.0-flash-001",
    "google/gemini-2.0-flash-001",
]

MODEL_OPTIONS = [
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "openai/gpt-5-mini",
    "openai/gpt-5",
    "openai/gpt-5.4-mini",
    "openai/gpt-5.4",
    "anthropic/claude-3.5-haiku",
    "anthropic/claude-3.7-sonnet",
    "anthropic/claude-haiku-4.5",
    "anthropic/claude-sonnet-4.6",
    "anthropic/claude-opus-4.6",
    "google/gemini-2.0-flash-001",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
    "meta-llama/llama-3.3-70b-instruct",
    "mistralai/mistral-large",
]

DEFAULT_CONSTRUCT_DEFINITION = (
    "Empathy is the degree to which the chatbot response demonstrates social "
    "perspective-taking and understanding of the customer's motivations and "
    "feelings, and uses that understanding to provide reassurance and personalized "
    "help.\n\n"
    'Likert 5-point scale (agreement with statement: "This response is empathetic '
    'for this persona/customer.")\n'
    "1 = Strongly disagree\n"
    "2 = Disagree\n"
    "3 = Neither agree nor disagree\n"
    "4 = Agree\n"
    "5 = Strongly agree"
)
DEFAULT_ROLE = (
    "Recreational rider. Moderate biking knowledge. Wants clear, simple, "
    "reassuring guidance. Low tolerance for jargon unless explained. Prefers "
    "step-by-step and practical recommendations."
)
DEFAULT_TASK = (
    "You will rate aspects (specified in Construct Definition) of a support "
    "chatbot response from the perspective of a specified rider persona on "
    "5-point Likert scale."
)
DEFAULT_INCLUSION_CRITERIA = (
    "Increase empathy when the response:\n"
    "- Is reassuring\n"
    "- Adopts the customer's point of view to consider the problem\n"
    "- Understands individual needs of the customer\n"
    "- Gives tailored options rather than one-size-fits-all to meet the individual "
    "needs of the customer\n"
    "- Discovers customer's personalized demands in time"
)
DEFAULT_EXCLUSION_CRITERIA = (
    "Decrease empathy when the response:\n"
    "- Is generic (same answer could fit any customer) without linking to the "
    "customer's stated needs\n"
    "- Ignores expressed discomfort, constraints, or goals\n"
    "- Provides no personalization\n"
    "- Has a non-reassuring tone (is cold, dismissive or salesy)\n"
    "- Uses \"efficiency-only\" behavior (is standardized, lacking elaboration "
    "where needed)"
)
DEFAULT_CONSTRAINTS = (
    "- return a flat JSON object only\n"
    '- for each scored dimension, use keys named "<dimension>_score" and "<dimension>_reasoning"\n'
    "- each *_score value must be an integer from 1 to 5\n"
    "- each *_reasoning value must be 1-2 short sentences\n"
    "- optional aggregate fields are allowed (for example average_empathy_score)\n"
    "- output valid JSON only\n"
    "- do not return markdown, code fences, or extra text"
)
DEFAULT_EXAMPLES = (
    "Example output:\n"
    '{\n'
    '  "individual_attention_score": 2,\n'
    '  "individual_attention_reasoning": "Some viewpoint alignment but weak personalization.",\n'
    '  "caring_tone_score": 3,\n'
    '  "caring_tone_reasoning": "The tone is somewhat reassuring but not especially warm.",\n'
    '  "average_empathy_score": 2.5\n'
    '}'
)
BLANK_PROMPT_VALUE = ""
DEFAULT_JUDGE_CONSTRUCT_DEFINITION = (
    "Use the original construct definition and scoring rubric being audited as "
    "context. Treat it as the target instrument whose wording, thresholds, and "
    "examples may need refinement to reduce disagreement between raters."
)
DEFAULT_JUDGE_ROLE = (
    "You are a measurement-audit judge. You do not score the item yourself. "
    "You analyze disagreements between raters and recommend edits to the "
    "rating instrument to improve Krippendorff's alpha."
)
DEFAULT_JUDGE_TASK = (
    "- Understand the construct definition text.\n"
    "- Read through the dataset, comparing the score and the reasoning values row by row.\n"
    "- Identify the source of disagreement by referring to the reasonings provided.\n"
    "- Based on your findings on the disagreements, suggest concrete refinements to the prompt, criteria, or examples that would reduce ambiguity and standardize interpretation of the construct."
)
DEFAULT_JUDGE_INCLUSION_CRITERIA = (
    "Propose edits when you detect any of the following:\n"
    "- Unclear construct definition\n"
    "- Ambiguous thresholds for score levels\n"
    "- Inconsistent interpretation of customer persona-specific standards\n"
    "- Missing examples for important edge cases\n"
    "- Conflicting rules within the instrument\n"
    "- Disagreement caused by vague wording in the criteria"
)
DEFAULT_JUDGE_EXCLUSION_CRITERIA = (
    "Do not propose edits that:\n"
    "- Redefine the construct into a different concept\n"
    "- Make the prompt longer unless the added text clearly reduces ambiguity\n"
    "- Judge the row directly instead of diagnosing disagreement"
)
DEFAULT_JUDGE_CONSTRAINTS = (
    "- Do not rate the item yourself\n"
    "- Focus on why the raters disagreed\n"
    "- Base recommendations only on the construct definition and rater outputs provided\n"
    "- Suggest concrete wording changes, not vague advice\n"
    "- Return valid JSON only\n"
    "- Do not return markdown, code fences, or extra text"
)
DEFAULT_JUDGE_EXAMPLES = (
    "Example recommendations:\n"
    "- Clarify that clarity should be judged independently from correctness\n"
    "- Define \"too vague\" as using placeholders without naming a concrete recommendation\n"
    "- Add an edge-case example where a response is concise but still unclear"
)


def build_system_prompt() -> str:
    system_sections = [
        ("Role", role_prompt.strip()),
        ("Task", task_prompt.strip()),
        ("Construct Definition", construct_definition.strip()),
        ("Inclusion Criteria", inclusion_criteria.strip()),
        ("Exclusion Criteria", exclusion_criteria.strip()),
        ("Constraints", constraints_prompt.strip()),
        ("Examples", examples_prompt.strip()),
        (
            "Output Format",
            (
                "Return valid JSON only as a flat object. For each scored dimension, "
                'use exactly two keys named "<dimension>_score" and '
                '"<dimension>_reasoning". Each "<dimension>_score" must be an integer '
                'from 1 to 5. Each "<dimension>_reasoning" must be 1-2 sentences. '
                "Optional aggregate scalar fields are allowed. Do not return markdown, "
                "code fences, or extra text."
            ),
        ),
    ]
    return "\n\n".join(f"{label}:\n{value}" for label, value in system_sections if value)


def build_judge_system_prompt() -> str:
    system_sections = [
        ("Construct Definition (Context)", judge_construct_definition.strip()),
        ("Role", judge_role_prompt.strip()),
        ("Task", judge_task_prompt.strip()),
        ("Inclusion Criteria", judge_inclusion_criteria.strip()),
        ("Exclusion Criteria", judge_exclusion_criteria.strip()),
        ("Constraints", judge_constraints_prompt.strip()),
        ("Examples", judge_examples_prompt.strip()),
        (
            "Output Format",
            (
                "Return valid JSON with exactly three top-level keys: "
                '"disagreement_summary", "sources_of_disagreement", and '
                '"recommended_edits". '
                '"disagreement_summary" must be a string. '
                '"sources_of_disagreement" must be an array of objects with exactly '
                'two keys: "issue" and "evidence". '
                '"recommended_edits" must be an array of objects with exactly four '
                'keys: "target_section", "problem", "proposed_replacement", and '
                '"rationale". '
                '"target_section" must be one of: "construct_definition", "task", '
                '"inclusion_criteria", "exclusion_criteria", "constraints", '
                'or "examples". Base your diagnosis only on the provided construct '
                "definition and rater outputs. Do not score rows yourself. Do not "
                "include any keys other than those specified."
            ),
        ),
    ]
    return "\n\n".join(f"{label}:\n{value}" for label, value in system_sections if value)


def normalize_cell_value(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_likert_score(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric_value = float(value)
        if numeric_value.is_integer() and 1 <= int(numeric_value) <= 5:
            return int(numeric_value)

    text = str(value).strip()
    if not text:
        return None

    if text in {"1", "2", "3", "4", "5"}:
        return int(text)

    try:
        numeric_value = float(text)
        if numeric_value.is_integer() and 1 <= int(numeric_value) <= 5:
            return int(numeric_value)
    except ValueError:
        pass

    return None


def extract_structured_result(raw_output: str) -> dict[str, object]:
    text = raw_output.strip()
    if not text:
        return {}

    fenced_match = re.search(r"```(?:\w+)?\s*(.*?)```", raw_output, flags=re.DOTALL)
    if fenced_match:
        text = fenced_match.group(1).strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            flattened: dict[str, object] = {}
            for key, value in parsed.items():
                if value is None:
                    continue
                normalized_key = str(key).strip()
                if not normalized_key:
                    continue
                if isinstance(value, (dict, list)):
                    flattened[normalized_key] = json.dumps(value, ensure_ascii=True)
                elif isinstance(value, (str, int, float, bool)):
                    flattened[normalized_key] = value
                else:
                    flattened[normalized_key] = str(value)
            return flattened
        if isinstance(parsed, (str, int, float)):
            return {"score": str(parsed).strip()}
    except Exception:
        pass

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return {}
    if len(lines) == 1:
        return {"score": lines[0].strip().strip('"').strip("'")}

    return {
        "score": lines[0].strip().strip('"').strip("'"),
        "reasoning": " ".join(lines[1:]).strip(),
    }


def normalize_output_field_name(field_name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", str(field_name).strip()).strip("_")
    return normalized.lower() or "value"


def normalize_output_payload(payload: dict[str, object]) -> dict[str, object]:
    normalized_payload: dict[str, object] = {}
    for key, value in payload.items():
        normalized_payload[normalize_output_field_name(key)] = value
    return normalized_payload


def build_model_column_prefix(model_name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", model_name).strip("_")
    return normalized.lower() or "model"


def collect_model_metric_sets(model_fields: dict[str, set[str]]) -> tuple[list[str], list[str]]:
    if not model_fields:
        return [], []

    comparable_score_fields = sorted(
        set.intersection(
            *[
                {
                    field[: -len("_score")]
                    for field in fields
                    if field.endswith("_score")
                    and f"{field[: -len('_score')]}_reasoning" in fields
                }
                for fields in model_fields.values()
            ]
        )
    )
    paired_field_names = {
        field_name
        for dimension_name in comparable_score_fields
        for field_name in (f"{dimension_name}_score", f"{dimension_name}_reasoning")
    }
    aggregate_fields = sorted(
        set.union(
            *[
                {
                    field
                    for field in fields
                    if field not in paired_field_names
                }
                for fields in model_fields.values()
            ]
        )
    )
    return comparable_score_fields, aggregate_fields


def build_alpha_input_dataframe(
    results_df: pd.DataFrame, model_prefixes: list[str], comparable_score_fields: list[str]
) -> pd.DataFrame:
    alpha_rows = []
    for _, row in results_df.iterrows():
        for field_name in comparable_score_fields:
            alpha_row = {}
            for model_prefix in model_prefixes:
                column_name = f"{model_prefix}__{field_name}_score"
                if column_name in results_df.columns:
                    alpha_row[model_prefix] = row.get(column_name)
            if len(alpha_row) >= 2:
                alpha_rows.append(alpha_row)
    return pd.DataFrame(alpha_rows)


def calculate_nominal_krippendorff_alpha(ratings_df: pd.DataFrame) -> float | None:
    pair_counts: dict[tuple[str, str], int] = {}
    value_counts: dict[str, int] = {}
    total_pairable_rows = 0

    for _, row in ratings_df.iterrows():
        row_values = []
        for value in row.tolist():
            if pd.notna(value) and str(value).strip():
                row_values.append(str(value).strip())

        if len(row_values) < 2:
            continue

        total_pairable_rows += 1
        for value in row_values:
            value_counts[value] = value_counts.get(value, 0) + 1

        for left_index in range(len(row_values)):
            for right_index in range(left_index + 1, len(row_values)):
                pair = tuple(sorted((row_values[left_index], row_values[right_index])))
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

    if total_pairable_rows == 0:
        return None

    disagreements = 0
    total_pairs = 0
    for pair, count in pair_counts.items():
        total_pairs += count
        if pair[0] != pair[1]:
            disagreements += count

    if total_pairs == 0:
        return None

    observed_disagreement = disagreements / total_pairs
    total_values = sum(value_counts.values())
    if total_values <= 1:
        return None

    expected_agreement = sum(
        (count / total_values) ** 2 for count in value_counts.values()
    )
    expected_disagreement = 1 - expected_agreement

    if math.isclose(expected_disagreement, 0.0):
        return 1.0 if math.isclose(observed_disagreement, 0.0) else None

    return 1 - (observed_disagreement / expected_disagreement)


def validate_step_4_input_columns(
    uploaded_df: pd.DataFrame,
) -> tuple[list[str], list[str], list[str], list[str]]:
    required_base_columns = ["row_id", "context", "content"]
    missing_columns = [
        column for column in required_base_columns if column not in uploaded_df.columns
    ]

    model_fields: dict[str, set[str]] = {}
    for column in uploaded_df.columns:
        if "__" not in column:
            continue
        model_prefix, field_name = column.split("__", 1)
        if not model_prefix or not field_name:
            continue
        model_fields.setdefault(model_prefix, set()).add(field_name)

    if not model_fields:
        score_prefixes = {
            column[: -len("_score")]
            for column in uploaded_df.columns
            if column.endswith("_score")
        }
        reasoning_prefixes = {
            column[: -len("_reasoning")]
            for column in uploaded_df.columns
            if column.endswith("_reasoning")
        }
        error_prefixes = {
            column[: -len("_error")]
            for column in uploaded_df.columns
            if column.endswith("_error")
        }
        model_prefixes = sorted(score_prefixes | reasoning_prefixes | error_prefixes)
        if len(model_prefixes) < 2:
            missing_columns.append(
                "At least two model output sets using legacy <model>_score columns or new <model>__<field> columns"
            )
        for prefix in model_prefixes:
            for suffix in ("_score", "_reasoning", "_error"):
                column_name = f"{prefix}{suffix}"
                if column_name not in uploaded_df.columns:
                    missing_columns.append(column_name)

        score_columns = [
            f"{prefix}_score" for prefix in model_prefixes if f"{prefix}_score" in uploaded_df.columns
        ]
        supporting_columns = []
        supporting_columns.extend(
            f"{prefix}_reasoning"
            for prefix in model_prefixes
            if f"{prefix}_reasoning" in uploaded_df.columns
        )
        supporting_columns.extend(
            f"{prefix}_error" for prefix in model_prefixes if f"{prefix}_error" in uploaded_df.columns
        )
        return missing_columns, score_columns, supporting_columns, model_prefixes

    model_prefixes = sorted(model_fields)
    if len(model_prefixes) < 2:
        missing_columns.append(
            "At least two model output sets using <model>__<field> columns"
        )

    for prefix in model_prefixes:
        if "error" not in model_fields.get(prefix, set()):
            missing_columns.append(f"{prefix}__error")

    comparable_score_fields, aggregate_fields = collect_model_metric_sets(model_fields)
    if not comparable_score_fields:
        missing_columns.append(
            'At least one shared "<dimension>_score" + "<dimension>_reasoning" pair across models'
        )

    score_columns = [
        f"{prefix}__{field_name}_score"
        for field_name in comparable_score_fields
        for prefix in model_prefixes
        if f"{prefix}__{field_name}_score" in uploaded_df.columns
    ]
    supporting_columns = [
        f"{prefix}__{field_name}_reasoning"
        for field_name in comparable_score_fields
        for prefix in model_prefixes
        if f"{prefix}__{field_name}_reasoning" in uploaded_df.columns
    ]
    supporting_columns.extend(
        f"{prefix}__{field_name}"
        for field_name in aggregate_fields
        for prefix in model_prefixes
        if f"{prefix}__{field_name}" in uploaded_df.columns
    )
    supporting_columns.extend(
        f"{prefix}__error" for prefix in model_prefixes if f"{prefix}__error" in uploaded_df.columns
    )

    return missing_columns, score_columns, supporting_columns, model_prefixes


def build_step_4_disagreement_subset(
    uploaded_df: pd.DataFrame, score_columns: list[str]
) -> pd.DataFrame:
    disagreement_mask = []

    for _, row in uploaded_df.iterrows():
        valid_scores = []
        for column in score_columns:
            score_value = normalize_likert_score(row.get(column))
            if score_value is not None:
                valid_scores.append(score_value)

        disagreement_mask.append(
            len(valid_scores) >= 2 and len(set(valid_scores)) > 1
        )

    return uploaded_df.loc[disagreement_mask].copy()


def build_step_4_user_prompt(disagreement_df: pd.DataFrame) -> str:
    serialized_rows = disagreement_df.fillna("").to_dict(orient="records")
    return (
        "Analyze the disagreement rows from the Step 3 output dataset. "
        "Identify why raters disagreed and recommend wording edits to the rating "
        "instrument.\n\n"
        f"Rows for analysis ({len(serialized_rows)}):\n"
        f"{json.dumps(serialized_rows, indent=2)}"
    )


def parse_json_response(raw_output: str) -> dict:
    text = raw_output.strip()
    if not text:
        raise ValueError("Model returned an empty response.")

    fenced_match = re.search(r"```(?:\w+)?\s*(.*?)```", raw_output, flags=re.DOTALL)
    if fenced_match:
        text = fenced_match.group(1).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        json_start = text.find("{")
        json_end = text.rfind("}")
        if json_start == -1 or json_end == -1 or json_end <= json_start:
            raise

        candidate = text[json_start : json_end + 1]
        sanitized_candidate = sanitize_json_text(candidate)
        parsed = json.loads(sanitized_candidate)

    if not isinstance(parsed, dict):
        raise ValueError("Model response was not a JSON object.")

    return parsed


def sanitize_json_text(text: str) -> str:
    sanitized_chars = []
    in_string = False
    escape_next = False

    for char in text:
        if escape_next:
            sanitized_chars.append(char)
            escape_next = False
            continue

        if char == "\\":
            sanitized_chars.append(char)
            escape_next = True
            continue

        if char == '"':
            sanitized_chars.append(char)
            in_string = not in_string
            continue

        if in_string:
            if char == "\n":
                sanitized_chars.append("\\n")
                continue
            if char == "\r":
                sanitized_chars.append("\\r")
                continue
            if char == "\t":
                sanitized_chars.append("\\t")
                continue
            if ord(char) < 32:
                sanitized_chars.append(f"\\u{ord(char):04x}")
                continue

        sanitized_chars.append(char)

    return "".join(sanitized_chars)


def validate_judge_output(parsed_output: dict) -> dict:
    required_keys = [
        "disagreement_summary",
        "sources_of_disagreement",
        "recommended_edits",
    ]
    missing_keys = [key for key in required_keys if key not in parsed_output]
    if missing_keys:
        raise ValueError(f"Missing required keys: {', '.join(missing_keys)}")

    if not isinstance(parsed_output["disagreement_summary"], str):
        raise ValueError('"disagreement_summary" must be a string.')
    if not isinstance(parsed_output["sources_of_disagreement"], list):
        raise ValueError('"sources_of_disagreement" must be an array.')
    if not isinstance(parsed_output["recommended_edits"], list):
        raise ValueError('"recommended_edits" must be an array.')

    return parsed_output


def invalidate_step_1_state() -> None:
    st.session_state["customerbot_step_1_saved"] = False
    st.session_state["customerbot_step_3_results"] = None


def invalidate_step_2_state() -> None:
    st.session_state["customerbot_step_2_saved"] = False
    st.session_state["customerbot_step_3_results"] = None


st.set_page_config(page_title="Customerbot", layout="wide")

if "customerbot_step_1_saved" not in st.session_state:
    st.session_state["customerbot_step_1_saved"] = False
if "customerbot_step_2_saved" not in st.session_state:
    st.session_state["customerbot_step_2_saved"] = False
if "customerbot_step_3_results" not in st.session_state:
    st.session_state["customerbot_step_3_results"] = None

st.title("Customerbot")
st.caption("Run the same prompt against three language models in parallel.")
st.subheader("OpenRouter API Key")
st.caption("This API key is used by both Step 3 and Step 4.")
openrouter_api_key = st.text_input("OpenRouter API Key", type="password")

st.subheader("Step 1: Enter System Parameters")
parameter_columns = st.columns(2)
with parameter_columns[0]:
    primary_model = st.selectbox(
        "Primary LLM Model",
        MODEL_OPTIONS,
        index=MODEL_OPTIONS.index(DEFAULT_MODELS[0]),
        key="customerbot_primary_model",
        on_change=invalidate_step_1_state,
    )
with parameter_columns[1]:
    secondary_model = st.selectbox(
        "Secondary LLM Model",
        MODEL_OPTIONS,
        index=MODEL_OPTIONS.index(DEFAULT_MODELS[1]),
        key="customerbot_secondary_model",
        on_change=invalidate_step_1_state,
    )

upload_columns = st.columns(2)
with upload_columns[0]:
    uploaded_data = st.file_uploader(
        "Upload data (CSV)",
        type=["csv"],
        on_change=invalidate_step_1_state,
    )
with upload_columns[1]:
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.0,
        step=0.1,
        on_change=invalidate_step_1_state,
    )

if uploaded_data is not None:
    try:
        input_df = pd.read_csv(uploaded_data)
        st.caption(
            f"Loaded {len(input_df)} rows and {len(input_df.columns)} columns from CSV."
        )
    except Exception as exc:
        input_df = None
        st.error(f"Could not read CSV file: {exc}")
else:
    input_df = None

save_step_1 = st.button("Save", key="customerbot_save_step_1")
if save_step_1:
    if input_df is None:
        st.session_state["customerbot_step_1_saved"] = False
        st.error("Upload a valid CSV file before saving Step 1.")
    elif "context" not in input_df.columns or "content" not in input_df.columns:
        st.session_state["customerbot_step_1_saved"] = False
        st.error("Uploaded CSV must contain `context` and `content` columns.")
    else:
        st.session_state["customerbot_step_1_saved"] = True
        st.success("Step 1 saved successfully.")

st.subheader("Step 2: Enter Prompt")
role_prompt = st.text_area(
    "Role",
    value=DEFAULT_ROLE,
    height=90,
    on_change=invalidate_step_2_state,
)
task_prompt = st.text_area(
    "Task",
    value=DEFAULT_TASK,
    height=100,
    on_change=invalidate_step_2_state,
)
construct_definition = st.text_area(
    "Construct Definition",
    value=DEFAULT_CONSTRUCT_DEFINITION,
    height=100,
    on_change=invalidate_step_2_state,
)
inclusion_criteria = st.text_area(
    "Inclusion Criteria",
    value=DEFAULT_INCLUSION_CRITERIA,
    height=100,
    on_change=invalidate_step_2_state,
)
exclusion_criteria = st.text_area(
    "Exclusion Criteria",
    value=DEFAULT_EXCLUSION_CRITERIA,
    height=100,
    on_change=invalidate_step_2_state,
)
constraints_prompt = st.text_area(
    "Constraints",
    value=DEFAULT_CONSTRAINTS,
    height=120,
    on_change=invalidate_step_2_state,
)
examples_prompt = st.text_area(
    "Examples",
    value=DEFAULT_EXAMPLES,
    height=120,
    on_change=invalidate_step_2_state,
)

save_step_2 = st.button("Save", key="customerbot_save_step_2")
if save_step_2:
    st.session_state["customerbot_step_2_saved"] = True
    st.success("Step 2 saved successfully.")

st.subheader("Step 3: Generate Data")
run_generation = st.button("Generate Synthetic Customer Data", type="primary")

if run_generation:
    if not openrouter_api_key:
        st.error("OpenRouter API key is required.")
    elif primary_model == secondary_model:
        st.error("Primary and secondary models must be different.")
    elif input_df is None:
        st.error("Upload a valid CSV file.")
    elif not st.session_state["customerbot_step_1_saved"]:
        st.error("Save Step 1 before generating data.")
    elif not st.session_state["customerbot_step_2_saved"]:
        st.error("Save Step 2 before generating data.")
    elif "context" not in input_df.columns or "content" not in input_df.columns:
        st.error("CSV must contain `context` and `content` columns.")
    elif not task_prompt.strip():
        st.error("Task is required.")
    else:
        selected_models = [primary_model, secondary_model]
        system_prompt = build_system_prompt()
        dataset_rows = []
        jobs = []

        for row_index, row in input_df.iterrows():
            original_row_id = row.get("row_id")
            row_id_value = original_row_id if pd.notna(original_row_id) else row_index
            context_value = normalize_cell_value(row.get("context", ""))
            content_value = normalize_cell_value(row.get("content", ""))
            dataset_rows.append(
                {
                    "row_id": row_id_value,
                    "context": context_value,
                    "content": content_value,
                }
            )

            row_user_prompt = (
                "Use the following fields from the uploaded CSV row to produce the requested ratings.\n\n"
                f"context:\n{context_value or '[empty]'}\n\n"
                f"content:\n{content_value or '[empty]'}"
            )
            for model in selected_models:
                jobs.append(
                    {
                        "row_id": row_id_value,
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": row_user_prompt},
                        ],
                    }
                )

        with st.spinner("Scoring dataset rows with both models..."):
            scored_outputs: dict[tuple[int, str], dict] = {}
            max_workers = min(4, max(1, len(jobs)))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {
                    executor.submit(
                        timed_openrouter_chat_completion,
                        api_key=openrouter_api_key,
                        model=job["model"],
                        messages=job["messages"],
                        temperature=temperature,
                    ): (job["row_id"], job["model"])
                    for job in jobs
                }
                for future in as_completed(future_map):
                    row_id, model = future_map[future]
                    result = future.result()
                    result["parsed_output"] = normalize_output_payload(
                        extract_structured_result(result["content"])
                    )
                    scored_outputs[(row_id, model)] = result

        model_prefix_map = {
            model: build_model_column_prefix(model) for model in selected_models
        }
        model_fields = {
            model: {
                field_name
                for row_id in [row["row_id"] for row in dataset_rows]
                for field_name in scored_outputs[(row_id, model)]["parsed_output"].keys()
            }
            for model in selected_models
        }
        comparable_score_fields, aggregate_fields = collect_model_metric_sets(model_fields)

        records = []
        for row_data in dataset_rows:
            row_id = row_data["row_id"]
            record = {
                "row_id": row_id,
                "context": row_data["context"],
                "content": row_data["content"],
            }
            for model in selected_models:
                model_prefix = model_prefix_map[model]
                model_result = scored_outputs[(row_id, model)]
                parsed_output = model_result["parsed_output"]
                paired_dimension_fields = {
                    f"{field_name}_score" for field_name in comparable_score_fields
                } | {
                    f"{field_name}_reasoning" for field_name in comparable_score_fields
                }

                for field_name, field_value in parsed_output.items():
                    column_name = f"{model_prefix}__{field_name}"
                    if field_name in paired_dimension_fields and field_name.endswith("_score"):
                        record[column_name] = normalize_likert_score(field_value)
                    elif field_name.endswith("_reasoning"):
                        record[column_name] = normalize_cell_value(field_value)
                    else:
                        record[column_name] = field_value

                record[f"{model_prefix}__error"] = model_result["error"]

            records.append(record)

        results_df = pd.DataFrame(records)
        alpha_input_df = build_alpha_input_dataframe(
            results_df,
            [model_prefix_map[model] for model in selected_models],
            comparable_score_fields,
        ).dropna()
        alpha = calculate_nominal_krippendorff_alpha(alpha_input_df)

        summary_df = pd.DataFrame(
            [
                {
                    "model": model,
                    "model_column_prefix": model_prefix_map[model],
                    "rows_scored": len(dataset_rows),
                    "successful_scores": sum(
                        1
                        for row_id in results_df["row_id"]
                        if not scored_outputs[(row_id, model)]["error"]
                    ),
                    "scored_dimensions": len(comparable_score_fields),
                    "average_duration_seconds": round(
                        sum(scored_outputs[(row_id, model)]["duration_seconds"] for row_id in results_df["row_id"])
                        / max(len(dataset_rows), 1),
                        2,
                    ),
                    "average_score_across_dimensions": round(
                        pd.concat(
                            [
                                results_df[f"{model_prefix_map[model]}__{field_name}_score"].dropna()
                                for field_name in comparable_score_fields
                                if f"{model_prefix_map[model]}__{field_name}_score" in results_df.columns
                            ],
                            ignore_index=True,
                        ).mean(),
                        2,
                    )
                    if comparable_score_fields
                    else None,
                    "median_score_across_dimensions": float(
                        pd.concat(
                            [
                                results_df[f"{model_prefix_map[model]}__{field_name}_score"].dropna()
                                for field_name in comparable_score_fields
                                if f"{model_prefix_map[model]}__{field_name}_score" in results_df.columns
                            ],
                            ignore_index=True,
                        ).median()
                    )
                    if comparable_score_fields
                    else None,
                }
                for model in selected_models
            ]
        )

        error_count = int(
            results_df[f"{model_prefix_map[primary_model]}__error"].astype(bool).sum()
        ) + int(
            results_df[f"{model_prefix_map[secondary_model]}__error"].astype(bool).sum()
        )

        export_payload = {
            "system_prompt": system_prompt.strip(),
            "prompt_sections": {
                "construct_definition": construct_definition.strip(),
                "role": role_prompt.strip(),
                "task": task_prompt.strip(),
                "inclusion_criteria": inclusion_criteria.strip(),
                "exclusion_criteria": exclusion_criteria.strip(),
                "constraints": constraints_prompt.strip(),
                "examples": examples_prompt.strip(),
            },
            "temperature": temperature,
            "krippendorff_alpha": alpha,
            "model_column_prefixes": model_prefix_map,
            "scored_dimensions": comparable_score_fields,
            "aggregate_fields": aggregate_fields,
            "results": records,
        }
        st.session_state["customerbot_step_3_results"] = {
            "summary_df": summary_df,
            "results_df": results_df,
            "alpha": alpha,
            "error_count": error_count,
            "export_payload": export_payload,
        }

step_3_results = st.session_state.get("customerbot_step_3_results")
if step_3_results:
    st.subheader("Run Summary")
    st.dataframe(step_3_results["summary_df"], use_container_width=True)

    st.subheader("Scored Dataset")
    st.dataframe(step_3_results["results_df"], use_container_width=True)
    st.subheader("Preview")
    st.dataframe(step_3_results["results_df"].head(10), use_container_width=True)

    st.metric(
        "Krippendorff's Alpha",
        "N/A" if step_3_results["alpha"] is None else f"{step_3_results['alpha']:.4f}",
    )

    if step_3_results["error_count"]:
        st.warning(f"{step_3_results['error_count']} model scoring calls returned errors.")

    st.download_button(
        "Download Results JSON",
        data=json.dumps(step_3_results["export_payload"], indent=2),
        file_name="customerbot-results.json",
        mime="application/json",
    )
    st.download_button(
        "Download Results CSV",
        data=step_3_results["results_df"].to_csv(index=False),
        file_name="customerbot-results.csv",
        mime="text/csv",
    )

st.subheader("Step 4: Optimize Reasoning")
judge_model = st.selectbox(
    "Judge LLM Model",
    MODEL_OPTIONS,
    index=MODEL_OPTIONS.index(DEFAULT_MODELS[0]),
    key="customerbot_judge_model",
)

step_4_columns = st.columns(2)
with step_4_columns[0]:
    uploaded_judge_data = st.file_uploader(
        "Upload Data (CSV)",
        type=["csv"],
        key="customerbot_step_4_upload",
    )
with step_4_columns[1]:
    judge_temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.0,
        step=0.1,
        key="customerbot_step_4_temperature",
    )

step_4_df = None
step_4_score_columns: list[str] = []
step_4_supporting_columns: list[str] = []
step_4_model_prefixes: list[str] = []
step_4_disagreement_df = pd.DataFrame()

if uploaded_judge_data is not None:
    try:
        step_4_df = pd.read_csv(uploaded_judge_data)
        st.caption(
            f"Loaded {len(step_4_df)} rows and {len(step_4_df.columns)} columns for Step 4."
        )
        (
            step_4_missing_columns,
            step_4_score_columns,
            step_4_supporting_columns,
            step_4_model_prefixes,
        ) = validate_step_4_input_columns(step_4_df)
        if step_4_missing_columns:
            step_4_df = None
            st.error(
                "Step 4 requires a Step 3 output CSV with these columns present: "
                + ", ".join(step_4_missing_columns)
            )
        else:
            step_4_disagreement_df = build_step_4_disagreement_subset(
                step_4_df, step_4_score_columns
            )
            st.caption(
                "Detected Step 3 output columns: "
                + ", ".join(["row_id", "context", "content"] + step_4_score_columns + step_4_supporting_columns)
            )
            st.caption("Detected model output groups: " + ", ".join(step_4_model_prefixes))
            st.caption(
                f"Disagreement rows available for Step 4 judge review: {len(step_4_disagreement_df)} of {len(step_4_df)}"
            )
            if not step_4_disagreement_df.empty:
                preview_columns = ["row_id", "context", "content"] + step_4_score_columns
                st.dataframe(
                    step_4_disagreement_df[preview_columns].head(10),
                    use_container_width=True,
                )
    except Exception as exc:
        step_4_df = None
        st.error(f"Could not read Step 4 CSV file: {exc}")
else:
    step_4_df = None

st.subheader("Prompt")
judge_construct_definition = st.text_area(
    "Construct Definition (Context)",
    value=DEFAULT_JUDGE_CONSTRUCT_DEFINITION,
    height=100,
    key="customerbot_step_4_construct_definition",
)
judge_role_prompt = st.text_area(
    "Role",
    value=DEFAULT_JUDGE_ROLE,
    height=90,
    key="customerbot_step_4_role",
)
judge_task_prompt = st.text_area(
    "Task",
    value=DEFAULT_JUDGE_TASK,
    height=100,
    key="customerbot_step_4_task",
)
judge_inclusion_criteria = st.text_area(
    "Inclusion Criteria",
    value=DEFAULT_JUDGE_INCLUSION_CRITERIA,
    height=100,
    key="customerbot_step_4_inclusion",
)
judge_exclusion_criteria = st.text_area(
    "Exclusion Criteria",
    value=DEFAULT_JUDGE_EXCLUSION_CRITERIA,
    height=100,
    key="customerbot_step_4_exclusion",
)
judge_constraints_prompt = st.text_area(
    "Constraints",
    value=DEFAULT_JUDGE_CONSTRAINTS,
    height=120,
    key="customerbot_step_4_constraints",
)
judge_examples_prompt = st.text_area(
    "Examples",
    value=DEFAULT_JUDGE_EXAMPLES,
    height=120,
    key="customerbot_step_4_examples",
)

run_step_4 = st.button("Run Step 4 Judge", type="primary")

if run_step_4:
    if not openrouter_api_key:
        st.error("OpenRouter API key is required.")
    elif step_4_df is None:
        st.error("Upload a valid Step 3 output CSV for Step 4.")
    elif step_4_disagreement_df.empty:
        st.error("No disagreement rows were found to send to the judge model.")
    else:
        judge_system_prompt = build_judge_system_prompt()
        judge_user_prompt = build_step_4_user_prompt(step_4_disagreement_df)

        with st.spinner("Running Step 4 judge model..."):
            judge_result = timed_openrouter_chat_completion(
                api_key=openrouter_api_key,
                model=judge_model,
                messages=[
                    {"role": "system", "content": judge_system_prompt},
                    {"role": "user", "content": judge_user_prompt},
                ],
                temperature=judge_temperature,
            )

        if judge_result["error"]:
            st.error(f"Step 4 judge call failed: {judge_result['error']}")
        else:
            try:
                parsed_judge_output = validate_judge_output(
                    parse_json_response(judge_result["content"])
                )
                st.subheader("Step 4 Judge Output")
                st.metric(
                    "Judge Duration (seconds)",
                    f"{judge_result['duration_seconds']:.2f}",
                )
                st.text_area(
                    "Disagreement Summary",
                    value=parsed_judge_output["disagreement_summary"],
                    height=140,
                )
                if parsed_judge_output["sources_of_disagreement"]:
                    st.subheader("Sources Of Disagreement")
                    st.dataframe(
                        pd.DataFrame(parsed_judge_output["sources_of_disagreement"]),
                        use_container_width=True,
                    )
                else:
                    st.caption("No disagreement sources were returned.")

                if parsed_judge_output["recommended_edits"]:
                    st.subheader("Recommended Edits")
                    st.dataframe(
                        pd.DataFrame(parsed_judge_output["recommended_edits"]),
                        use_container_width=True,
                    )
                else:
                    st.caption("No recommended edits were returned.")

                st.subheader("Raw JSON")
                st.json(parsed_judge_output)
                st.download_button(
                    "Download Step 4 Judge JSON",
                    data=json.dumps(parsed_judge_output, indent=2),
                    file_name="customerbot-step-4-judge.json",
                    mime="application/json",
                )
            except Exception as exc:
                st.error(f"Step 4 judge returned invalid JSON: {exc}")
                st.text_area(
                    "Raw Step 4 Judge Output",
                    value=judge_result["content"],
                    height=240,
                )
