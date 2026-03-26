import math

import pandas as pd
import streamlit as st


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
    col_marginals = [sum(observed[row_index][col_index] for row_index in range(num_values)) for col_index in range(num_values)]

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


def build_qwk_table(
    uploaded_df: pd.DataFrame, model_prefixes: list[str], field_names: list[str], schema_kind: str
) -> pd.DataFrame:
    records = []
    if len(model_prefixes) != 2:
        return pd.DataFrame(
            [
                {
                    "field": field_name,
                    "pairable_rows": 0,
                    "quadratic_weighted_kappa": None,
                }
                for field_name in field_names
            ]
        )

    for field_name in field_names:
        field_ratings_df = build_field_rating_frame(uploaded_df, model_prefixes, field_name, schema_kind)
        comparable_rows = int(len(field_ratings_df.dropna()))
        qwk = calculate_quadratic_weighted_kappa(field_ratings_df.dropna())
        records.append(
            {
                "field": field_name,
                "pairable_rows": comparable_rows,
                "quadratic_weighted_kappa": None if qwk is None else round(qwk, 4),
            }
        )

    return pd.DataFrame(records)


st.set_page_config(page_title="QWK Calculator", layout="wide")

st.title("Quadratic Weighted Kappa Calculator")
st.caption("Upload a Step 3 CSV and calculate QWK for each shared score pair and the shared average score pair.")

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
            elif len(model_prefixes) != 2:
                st.error("QWK requires exactly two model output groups in the uploaded CSV.")
            else:
                paired_dimensions, aggregate_score_fields = collect_shared_score_fields(model_fields)
                average_score_fields = [
                    field_name for field_name in aggregate_score_fields if "average" in field_name.lower()
                ]

                st.caption(f"Detected schema: {schema_kind}")
                st.caption("Detected model groups: " + ", ".join(model_prefixes))

                if paired_dimensions:
                    dimension_score_fields = [f"{dimension}_score" for dimension in paired_dimensions]
                    per_dimension_qwk_df = build_qwk_table(
                        uploaded_df, model_prefixes, dimension_score_fields, schema_kind
                    )
                    st.subheader("Per-Dimension QWK")
                    st.dataframe(per_dimension_qwk_df, use_container_width=True)
                else:
                    st.warning("No shared scored dimensions with matching reasoning fields were found.")

                if average_score_fields:
                    average_qwk_df = build_qwk_table(
                        uploaded_df, model_prefixes, average_score_fields, schema_kind
                    )
                    st.subheader("Average Score QWK")
                    st.dataframe(average_qwk_df, use_container_width=True)
                else:
                    st.info("No shared average score field was found.")
