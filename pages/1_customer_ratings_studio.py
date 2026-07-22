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
    "google/gemini-2.5-flash",
    "anthropic/claude-sonnet-4.6",
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
    "The construct you are rating should be defined here in plain language. "
    "Describe what a strong response looks like, what a weak response looks "
    "like, and how the 5-point scale should be interpreted.\n\n"
    'Likert 5-point scale (agreement with statement: "This response matches the '
    'defined construct for this case.")\n'
    "1 = Strongly disagree\n"
    "2 = Disagree\n"
    "3 = Neither agree nor disagree\n"
    "4 = Agree\n"
    "5 = Strongly agree"
)
DEFAULT_ROLE = (
    "You are a service quality evaluation expert. You are assessing sales "
    "support provided by the supporter to the customer."
)
DEFAULT_TASK = (
    "Refer to the context and rate content based on the items specified in "
    "Construct Definition. Rate each item on a 5-point Likert scale."
)
DEFAULT_INCLUSION_CRITERIA = (
    "individualized_attention includes:\n"
    "- Explicit references to customer's specific use case, preferences or needs\n"
    "caring_tone includes:\n"
    "- Explict use of reassuring vocabulary when the context contains emotional "
    "load such as pain, discomfort, frustration\n"
    "customer_first_orientation includes:\n"
    "- Appearance of product details and customer's requirements together\n"
    "need_understanding\n"
    "- Explicit acknowledgement of the need, preference or problem"
)
DEFAULT_EXCLUSION_CRITERIA = (
    "Do not penalize individualized_attention if:\n"
    "- The context does not have any details on the customer's specific use "
    "case, preferences or needs\n"
    "Do not penalize caring_tone if:\n"
    "- The context does not carry emotional load\n"
    "Do not penalize customer_first_orientation if:\n"
    "- The context does not have any details on the customer's specific use "
    "case, preferences or needs\n"
    "Do not penalize need_understanding if:\n"
    "- The context does not have any details on the customer's specific use "
    "case, preferences or needs"
)
DEFAULT_CONSTRAINTS = (
    "- Do not calculate average_empathy_score yourself.\n"
    "- Item Scores: integers 1-5\n"
    "- Item Reasoning: 1-2 short sentences per item\n"
    "- Output: valid JSON only, no markdown or extra text"
)
DEFAULT_EXAMPLES = (
    "Examples:\n"
    '{  "examples": [\n'
    '    {\n'
    '      "item": "individual_attention",\n'
    '      "definition": "The brand employees give customers individual attention",\n'
    '      "example": {\n'
    '        "context": "I\'ll be riding in Whistler and the forecast says it might be rainy. What do you recommend?",\n'
    '        "content": "This grip has an aggressive finger traction pattern and ribbed palm that adds extra surface for you to hold on better."\n'
    '      },\n'
    '      "annotation": {\n'
    '        "what_happened": "The supporter fulfills the customer\'s informational need, but it is overly focused on listing the product specs instead of tying them back to the customer\'s intent. It appears like a generic sales pitch and lacks a personalized feel.",\n'
    '        "more_empathic_alternative": "This grip has an aggressive finger traction pattern designed to make you hold on better when wet. The ribbed palm adds extra surface for stronger grip. These features will help safer riding under rainy conditions in Whistler.",\n'
    '        "why_more_empathic": "This approach highlights product features while acknowledging the customer\'s specific use case. This makes the customer feel they are getting personalized recommendations."\n'
    '      }\n'
    '    },\n'
    '    {\n'
    '      "item": "caring_fashion",\n'
    '      "definition": "The brand employees deal with customers in a caring fashion",\n'
    '      "example": {\n'
    '        "context": "I\'m looking for grips that are durable. I don\'t want to spend too much money.",\n'
    '        "content": "This is a durable grip. It is also at an affordable price point at $12.99."\n'
    '      },\n'
    '      "annotation": {\n'
    '        "what_happened": "The supporter fulfills the customer\'s informational need, but is too direct and brief. This makes it appear uncaring and dismissive.",\n'
    '        "more_empathic_alternative": "Sure, I can help you find durable grips that are affordable at the same time.",\n'
    '        "why_more_empathic": "This approach fulfills the customer\'s informational need while also ensuring the customer knows they have the attention of the support rep."\n'
    '      }\n'
    '    },\n'
    '    {\n'
    '      "item": "best_interest_at_heart",\n'
    '      "definition": "The brand employees have the customer best interest at heart",\n'
    '      "example": {\n'
    '        "context": "Large grips are too bulky and they make me frustrated. But I have large hands, what can I do?",\n'
    '        "content": "This is a slim grip that suits large hands and offers padding without being bulky."\n'
    '      },\n'
    '      "annotation": {\n'
    '        "what_happened": "The supporter fulfills the customer\'s informational need and addresses their individual requirements, but it does not establish a personal connection. The customer asks an emotionally loaded question, expressing frustration, but the support does not acknowledge the emotion. It makes it look like support is prioritizing a sale rather than the customer\'s comfort.",\n'
    '        "more_empathic_alternative": "I understand bulkiness can be frustrating. Here is a slimmer alternative that offers high damping, and balances comfort and traction. This hybrid design helps keep riding enjoyable and comfortable for larger hands.",\n'
    '        "why_more_empathic": "This approach highlights product features while acknowledging the customer\'s emotional state and preferences. This makes the customer feel reassured that the support has their best interest at heart."\n'
    '      }\n'
    '    },\n'
    '    {\n'
    '      "item": "understand_customer_needs",\n'
    '      "definition": "The brand employees understand the needs of their customers",\n'
    '      "example": {\n'
    '        "context": "My hands hurt after a while with my current grips.",\n'
    '        "content": "This grip has thick padding."\n'
    '      },\n'
    '      "annotation": {\n'
    '        "what_happened": "The supporter makes a recommendation with a relevant feature, but it does not acknowledge the customer\'s specific need. This shows a lack of insight and understanding, looking like a generic RAG response.",\n'
    '        "more_empathic_alternative": "I understand you may need extra padding. This grip has a high damping level, which means thicker padding to reduce hand pain. This grip addresses your hand pain concern with a medium diameter and hybrid offset design that provides extra padding on the palm side, where you\'re experiencing discomfort between the thumb and pointer.",\n'
    '        "why_more_empathic": "This approach highlights product features while acknowledging the customer\'s specific need and how the product feature relates to the need. This indicates insight and initiative."\n'
    '      }\n'
    '    }\n'
    '  ]\n'
    '}'
)
BLANK_PROMPT_VALUE = ""
DEFAULT_JUDGE_CONSTRUCT_DEFINITION = (
    "Empathy is the degree to which the response makes the customer feel "
    "understood, cared for, and personally helped through its wording and "
    "tailoring.\n\n"
    "Empathy is composed of the below items:\n"
    "- individual_attention: The response refers to the customer's specific "
    "situation, constraints, or use case rather than giving a generic answer.\n"
    "- caring_tone: The response uses a supportive, reassuring, non-dismissive "
    "tone and acknowledges any discomfort, worry, or frustration expressed by "
    "the customer.\n"
    "- customer_first_orientation: The response prioritizes the customer's "
    "comfort, safety, fit, or stated goals rather than sounding salesy, "
    "scripted, or product-first.\n"
    "- need_understanding: The response accurately identifies the customer's "
    "stated need or problem and explains how the recommendation addresses it."
)
DEFAULT_JUDGE_ROLE = (
    "You are a measurement-audit judge. You do not score the item yourself. "
    "You analyze disagreements between raters and recommend edits to the "
    "rating instrument to improve quadratic weighted kappa."
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
    "- Clarify the edge cases where a response is short but still complete\n"
    "- Define what counts as too vague versus appropriately concise\n"
    "- Add an example showing a borderline score and why it belongs there"
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


def normalize_numeric_rating(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)

    text = str(value).strip()
    if not text:
        return None

    try:
        return float(text)
    except ValueError:
        return None


def build_qwk_input_dataframe(
    results_df: pd.DataFrame, model_prefixes: list[str], comparable_score_fields: list[str]
) -> pd.DataFrame:
    qwk_rows = []
    for _, row in results_df.iterrows():
        for field_name in comparable_score_fields:
            qwk_row = {}
            for model_prefix in model_prefixes:
                column_name = f"{model_prefix}__{field_name}_score"
                if column_name in results_df.columns:
                    qwk_row[model_prefix] = row.get(column_name)
            if len(qwk_row) >= 2:
                qwk_rows.append(qwk_row)
    return pd.DataFrame(qwk_rows)


def build_field_qwk_frame(
    results_df: pd.DataFrame, model_prefixes: list[str], field_name: str
) -> pd.DataFrame:
    qwk_rows = []
    for _, row in results_df.iterrows():
        qwk_row = {}
        for model_prefix in model_prefixes:
            column_name = f"{model_prefix}__{field_name}_score"
            if column_name in results_df.columns:
                qwk_row[model_prefix] = row.get(column_name)
        if len(qwk_row) >= 2:
            qwk_rows.append(qwk_row)
    return pd.DataFrame(qwk_rows)


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


def calculate_quadratic_weighted_kappa(ratings_df: pd.DataFrame) -> float | None:
    normalized_rows = []
    for _, row in ratings_df.iterrows():
        values = [normalize_numeric_rating(value) for value in row.tolist()]
        if any(value is None for value in values):
            continue
        if len(values) != 2:
            continue
        normalized_rows.append((values[0], values[1]))

    if not normalized_rows:
        return None

    unique_values = sorted({value for pair in normalized_rows for value in pair})
    if len(unique_values) <= 1:
        return 1.0

    value_to_index = {value: index for index, value in enumerate(unique_values)}
    num_values = len(unique_values)

    observed = [[0.0 for _ in range(num_values)] for _ in range(num_values)]
    for left_rating, right_rating in normalized_rows:
        observed[value_to_index[left_rating]][value_to_index[right_rating]] += 1.0

    total = float(len(normalized_rows))
    row_marginals = [sum(row) for row in observed]
    col_marginals = [
        sum(observed[row_index][col_index] for row_index in range(num_values))
        for col_index in range(num_values)
    ]

    expected = [
        [
            (row_marginals[row_index] * col_marginals[col_index]) / total
            for col_index in range(num_values)
        ]
        for row_index in range(num_values)
    ]

    denominator_base = float((num_values - 1) ** 2)
    if math.isclose(denominator_base, 0.0):
        return 1.0

    observed_weighted = 0.0
    expected_weighted = 0.0
    for row_index in range(num_values):
        for col_index in range(num_values):
            weight = ((row_index - col_index) ** 2) / denominator_base
            observed_weighted += weight * observed[row_index][col_index]
            expected_weighted += weight * expected[row_index][col_index]

    if math.isclose(expected_weighted, 0.0):
        return 1.0 if math.isclose(observed_weighted, 0.0) else None

    return 1 - (observed_weighted / expected_weighted)


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
        "Analyze the disagreement rows from the scored dataset. "
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


st.set_page_config(page_title="Customer Ratings Studio", layout="wide")

if "customerbot_step_1_saved" not in st.session_state:
    st.session_state["customerbot_step_1_saved"] = False
if "customerbot_step_2_saved" not in st.session_state:
    st.session_state["customerbot_step_2_saved"] = False
if "customerbot_step_3_results" not in st.session_state:
    st.session_state["customerbot_step_3_results"] = None

with st.sidebar:
    st.image("odi-logo.jpg", use_container_width=True)

st.title("Customer Ratings Studio")
st.caption(
    "Upload the Readability Evaluator output to simulate service quality ratings with LLMs."
)
st.subheader("API Key")
st.caption("This API key powers both scoring and review.")
openrouter_api_key = st.text_input("OpenRouter API Key", type="password")

st.subheader("Step 1: Prepare the run")
st.caption("Select the two LLMs that will rate the customer service transcript.")
parameter_columns = st.columns(2)
with parameter_columns[0]:
    primary_model = st.selectbox(
        "Primary rater",
        MODEL_OPTIONS,
        index=MODEL_OPTIONS.index(DEFAULT_MODELS[0]),
        key="customerbot_primary_model",
        on_change=invalidate_step_1_state,
    )
with parameter_columns[1]:
    secondary_model = st.selectbox(
        "Secondary rater",
        MODEL_OPTIONS,
        index=MODEL_OPTIONS.index(DEFAULT_MODELS[1]),
        key="customerbot_secondary_model",
        on_change=invalidate_step_1_state,
    )

upload_columns = st.columns(2)
with upload_columns[0]:
    uploaded_data = st.file_uploader(
        "Upload Readability Evaluator dataset (CSV)",
        type=["csv"],
        on_change=invalidate_step_1_state,
    )
with upload_columns[1]:
    temperature = st.slider(
        "Rater temperature",
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

save_step_1 = st.button("Save setup", key="customerbot_save_step_1")
if save_step_1:
    if input_df is None:
        st.session_state["customerbot_step_1_saved"] = False
        st.error("Upload a valid CSV file before saving the setup.")
    elif "context" not in input_df.columns or "content" not in input_df.columns:
        st.session_state["customerbot_step_1_saved"] = False
        st.error("Uploaded CSV must contain `context` and `content` columns.")
    else:
        st.session_state["customerbot_step_1_saved"] = True
        st.success("Setup saved successfully.")

st.subheader("Step 2: Define the rubric")
st.caption("Enter detailed instructions for the rater LLMs.")
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

save_step_2 = st.button("Save rubric", key="customerbot_save_step_2")
if save_step_2:
    st.session_state["customerbot_step_2_saved"] = True
    st.success("Rubric saved successfully.")

st.subheader("Step 3: Score the dataset")
st.caption("Run the LLMs to rate the customer service responses based on the rubric you defined in Step 2.")
run_generation = st.button("Run scoring", type="primary")

if run_generation:
    if not openrouter_api_key:
        st.error("OpenRouter API key is required.")
    elif primary_model == secondary_model:
        st.error("Primary and secondary models must be different.")
    elif input_df is None:
        st.error("Upload a valid CSV file.")
    elif not st.session_state["customerbot_step_1_saved"]:
        st.error("Save the setup before running scoring.")
    elif not st.session_state["customerbot_step_2_saved"]:
        st.error("Save the rubric before running scoring.")
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

        with st.spinner("Scoring dataset rows with both raters..."):
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
                score_values: list[float] = []

                for field_name, field_value in parsed_output.items():
                    column_name = f"{model_prefix}__{field_name}"
                    if field_name in paired_dimension_fields and field_name.endswith("_score"):
                        normalized_score = normalize_likert_score(field_value)
                        record[column_name] = normalized_score
                        if normalized_score is not None:
                            score_values.append(float(normalized_score))
                    elif field_name.endswith("_reasoning"):
                        record[column_name] = normalize_cell_value(field_value)
                    else:
                        record[column_name] = field_value

                if score_values and len(score_values) == len(comparable_score_fields):
                    record[f"{model_prefix}__average_empathy_score"] = round(
                        sum(score_values) / len(score_values),
                        2,
                    )
                else:
                    record.setdefault(f"{model_prefix}__average_empathy_score", None)

                record[f"{model_prefix}__error"] = model_result["error"]

            records.append(record)

        results_df = pd.DataFrame(records)
        alpha_input_df = build_alpha_input_dataframe(
            results_df,
            [model_prefix_map[model] for model in selected_models],
            comparable_score_fields,
        ).dropna()
        alpha = calculate_nominal_krippendorff_alpha(alpha_input_df)
        qwk_input_df = build_qwk_input_dataframe(
            results_df,
            [model_prefix_map[model] for model in selected_models],
            comparable_score_fields,
        ).dropna()
        qwk = calculate_quadratic_weighted_kappa(qwk_input_df)
        qwk_records = []
        for field_name in comparable_score_fields:
            field_qwk_df = build_field_qwk_frame(
                results_df,
                [model_prefix_map[model] for model in selected_models],
                field_name,
            ).dropna()
            field_qwk = calculate_quadratic_weighted_kappa(field_qwk_df)
            qwk_records.append(
                {
                    "dimension": field_name,
                    "pairable_rows": int(len(field_qwk_df)),
                    "quadratic_weighted_kappa": None if field_qwk is None else round(field_qwk, 4),
                    "meets_threshold": (
                        field_qwk is not None and field_qwk >= 0.6
                    ),
                }
            )
        qwk_table_df = pd.DataFrame(qwk_records)

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
            "quadratic_weighted_kappa": qwk,
            "quadratic_weighted_kappa_by_dimension": qwk_records,
            "model_column_prefixes": model_prefix_map,
            "scored_dimensions": comparable_score_fields,
            "aggregate_fields": aggregate_fields,
            "results": records,
        }
        st.session_state["customerbot_step_3_results"] = {
            "summary_df": summary_df,
            "results_df": results_df,
            "alpha": alpha,
            "qwk": qwk,
            "qwk_table_df": qwk_table_df,
            "error_count": error_count,
            "export_payload": export_payload,
        }

step_3_results = st.session_state.get("customerbot_step_3_results")
if step_3_results:
    st.subheader("Run Overview")
    st.dataframe(step_3_results["summary_df"], use_container_width=True)

    st.subheader("Scored Rows")
    st.dataframe(step_3_results["results_df"], use_container_width=True)
    st.subheader("Preview")
    st.dataframe(step_3_results["results_df"].head(10), use_container_width=True)

    st.metric(
        "Krippendorff's Alpha",
        "N/A" if step_3_results["alpha"] is None else f"{step_3_results['alpha']:.4f}",
    )
    st.metric(
        "Quadratic Weighted Kappa",
        "N/A" if step_3_results["qwk"] is None else f"{step_3_results['qwk']:.4f}",
    )

    st.subheader("Per-Dimension QWK")
    if not step_3_results["qwk_table_df"].empty:
        styled_qwk_df = (
            step_3_results["qwk_table_df"]
            .style.format({"quadratic_weighted_kappa": "{:.4f}"})
            .map(
                lambda value: "color: green; font-weight: 600;"
                if value is True
                else "",
                subset=["meets_threshold"],
            )
            .map(
                lambda value: "color: green; font-weight: 600;"
                if isinstance(value, (int, float)) and value >= 0.6
                else "",
                subset=["quadratic_weighted_kappa"],
            )
        )
        st.dataframe(styled_qwk_df, use_container_width=True)
    else:
        st.info("No shared scored dimensions were found for QWK calculation.")

    with st.expander("How to interpret these agreement metrics"):
        st.markdown(
            """
            - **Krippendorff's Alpha:**
              Measures how consistently the two raters assign scores across the
              dataset, accounting for agreement that could happen by chance.
              Values closer to `1.0` indicate stronger agreement. There is no
              hard threshold in this app, but higher is better.
            - **Quadratic Weighted Kappa (QWK):**
              Measures how closely the two raters agree on each ordinal score,
              while giving partial credit when ratings are close instead of far
              apart. Values closer to `1.0` indicate stronger agreement.
              Threshold: `>= 0.6`.
            - **Threshold meaning:**
              Any per-dimension QWK value at or above `0.6` is highlighted in
              green as a practical sign of acceptable agreement.
            """
        )

    if step_3_results["error_count"]:
        st.warning(f"{step_3_results['error_count']} model scoring calls returned errors.")

    st.download_button(
        "Download run JSON",
        data=json.dumps(step_3_results["export_payload"], indent=2),
        file_name="customerbot-results.json",
        mime="application/json",
    )
    st.download_button(
        "Download run CSV",
        data=step_3_results["results_df"].to_csv(index=False),
        file_name="customerbot-results.csv",
        mime="text/csv",
    )

st.subheader("Step 4: Review disagreements")
st.caption("Select the LLM that will review and compare the ratings from the LLMs you selected in Step 1.")
judge_model = st.selectbox(
    "Judge model",
    MODEL_OPTIONS,
    index=MODEL_OPTIONS.index(DEFAULT_MODELS[2]),
    key="customerbot_judge_model",
)

step_4_columns = st.columns(2)
with step_4_columns[0]:
    uploaded_judge_data = st.file_uploader(
        "Upload scored dataset (CSV)",
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
            f"Loaded {len(step_4_df)} rows and {len(step_4_df.columns)} columns for disagreement review."
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
                "This review step requires a scored CSV with these columns present: "
                + ", ".join(step_4_missing_columns)
            )
        else:
            step_4_disagreement_df = build_step_4_disagreement_subset(
                step_4_df, step_4_score_columns
            )
            st.caption(
                "Detected scored columns: "
                + ", ".join(["row_id", "context", "content"] + step_4_score_columns + step_4_supporting_columns)
            )
            st.caption("Detected model groups: " + ", ".join(step_4_model_prefixes))
            st.caption(
                f"Rows with disagreements ready for review: {len(step_4_disagreement_df)} of {len(step_4_df)}"
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
st.caption("Enter detailed instructions for the reviewer LLM.")
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

run_step_4 = st.button("Review disagreements", type="primary")

if run_step_4:
    if not openrouter_api_key:
        st.error("OpenRouter API key is required.")
    elif step_4_df is None:
        st.error("Upload a valid scored CSV for disagreement review.")
    elif step_4_disagreement_df.empty:
        st.error("No disagreement rows were found to send to the reviewer model.")
    else:
        judge_system_prompt = build_judge_system_prompt()
        judge_user_prompt = build_step_4_user_prompt(step_4_disagreement_df)

        with st.spinner("Reviewing disagreement rows..."):
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
            st.error(f"Review assistant call failed: {judge_result['error']}")
        else:
            try:
                parsed_judge_output = validate_judge_output(
                    parse_json_response(judge_result["content"])
                )
                st.subheader("Review Output")
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
                    st.subheader("Sources of disagreement")
                    st.dataframe(
                        pd.DataFrame(parsed_judge_output["sources_of_disagreement"]),
                        use_container_width=True,
                    )
                else:
                    st.caption("No disagreement sources were returned.")

                if parsed_judge_output["recommended_edits"]:
                    st.subheader("Recommended edits")
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
                st.error(f"Review assistant returned invalid JSON: {exc}")
                st.text_area(
                    "Raw Step 4 Judge Output",
                    value=judge_result["content"],
                    height=240,
                )
