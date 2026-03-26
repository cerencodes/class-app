import math

import pandas as pd
import streamlit as st


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

    expected_agreement = sum((count / total_values) ** 2 for count in value_counts.values())
    expected_disagreement = 1 - expected_agreement

    if math.isclose(expected_disagreement, 0.0):
        return 1.0 if math.isclose(observed_disagreement, 0.0) else None

    return 1 - (observed_disagreement / expected_disagreement)


def discover_model_fields(uploaded_df: pd.DataFrame) -> tuple[dict[str, set[str]], str]:
    dynamic_model_fields: dict[str, set[str]] = {}
    for column in uploaded_df.columns:
        if "__" not in column:
            continue
        model_prefix, field_name = column.split("__", 1)
        if model_prefix and field_name:
            dynamic_model_fields.setdefault(model_prefix, set()).add(field_name)

    if dynamic_model_fields:
        return dynamic_model_fields, "dynamic"

    legacy_model_fields: dict[str, set[str]] = {}
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

    for prefix in sorted(score_prefixes | reasoning_prefixes | error_prefixes):
        fields = set()
        if f"{prefix}_score" in uploaded_df.columns:
            fields.add("score")
        if f"{prefix}_reasoning" in uploaded_df.columns:
            fields.add("reasoning")
        if f"{prefix}_error" in uploaded_df.columns:
            fields.add("error")
        legacy_model_fields[prefix] = fields

    return legacy_model_fields, "legacy"


def collect_shared_score_fields(model_fields: dict[str, set[str]]) -> tuple[list[str], list[str]]:
    if not model_fields:
        return [], []

    shared_fields = sorted(set.intersection(*model_fields.values()))
    paired_dimensions = []
    aggregate_scores = []

    for field_name in shared_fields:
        if field_name.endswith("_score") and f"{field_name[: -len('_score')]}_reasoning" in shared_fields:
            paired_dimensions.append(field_name[: -len("_score")])
        elif field_name == "score":
            aggregate_scores.append(field_name)
        elif field_name.endswith("_score"):
            aggregate_scores.append(field_name)

    return sorted(set(paired_dimensions)), sorted(set(aggregate_scores))


def build_field_rating_frame(
    uploaded_df: pd.DataFrame, model_prefixes: list[str], field_name: str, schema_kind: str
) -> pd.DataFrame:
    ratings_rows = []
    for _, row in uploaded_df.iterrows():
        rating_row = {}
        for model_prefix in model_prefixes:
            if schema_kind == "dynamic":
                column_name = f"{model_prefix}__{field_name}"
            else:
                if field_name == "score":
                    column_name = f"{model_prefix}_score"
                else:
                    column_name = f"{model_prefix}_{field_name}"

            if column_name in uploaded_df.columns:
                rating_row[model_prefix] = row.get(column_name)

        if len(rating_row) >= 2:
            ratings_rows.append(rating_row)

    return pd.DataFrame(ratings_rows)


def build_alpha_table(
    uploaded_df: pd.DataFrame, model_prefixes: list[str], field_names: list[str], schema_kind: str
) -> pd.DataFrame:
    records = []
    for field_name in field_names:
        field_ratings_df = build_field_rating_frame(uploaded_df, model_prefixes, field_name, schema_kind)
        comparable_rows = int(len(field_ratings_df.dropna()))
        alpha = calculate_nominal_krippendorff_alpha(field_ratings_df.dropna())
        records.append(
            {
                "field": field_name,
                "pairable_rows": comparable_rows,
                "krippendorff_alpha": None if alpha is None else round(alpha, 4),
            }
        )

    return pd.DataFrame(records)


st.set_page_config(page_title="Alpha Calculator", layout="wide")

st.title("Krippendorff's Alpha Calculator")
st.caption("Upload a Step 3 CSV and calculate alpha for each shared score pair and the shared average score pair.")

uploaded_file = st.file_uploader("Upload Step 3 CSV", type=["csv"])

if uploaded_file is not None:
    try:
        uploaded_df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read CSV file: {exc}")
    else:
        required_base_columns = {"row_id", "context", "content"}
        missing_base_columns = sorted(required_base_columns - set(uploaded_df.columns))
        if missing_base_columns:
            st.error(
                "The uploaded CSV is missing required columns: "
                + ", ".join(missing_base_columns)
            )
        else:
            model_fields, schema_kind = discover_model_fields(uploaded_df)
            model_prefixes = sorted(model_fields)

            if len(model_prefixes) < 2:
                st.error("The uploaded CSV must contain at least two model output groups.")
            else:
                paired_dimensions, aggregate_score_fields = collect_shared_score_fields(model_fields)
                average_score_fields = [
                    field_name for field_name in aggregate_score_fields if "average" in field_name.lower()
                ]

                st.caption(f"Detected schema: {schema_kind}")
                st.caption("Detected model groups: " + ", ".join(model_prefixes))

                if paired_dimensions:
                    dimension_score_fields = [f"{dimension}_score" for dimension in paired_dimensions]
                    per_dimension_alpha_df = build_alpha_table(
                        uploaded_df, model_prefixes, dimension_score_fields, schema_kind
                    )
                    st.subheader("Per-Dimension Alpha")
                    st.dataframe(per_dimension_alpha_df, use_container_width=True)
                else:
                    st.warning("No shared scored dimensions with matching reasoning fields were found.")

                if average_score_fields:
                    average_alpha_df = build_alpha_table(
                        uploaded_df, model_prefixes, average_score_fields, schema_kind
                    )
                    st.subheader("Average Score Alpha")
                    st.dataframe(average_alpha_df, use_container_width=True)
                else:
                    st.info("No shared average score field was found.")
