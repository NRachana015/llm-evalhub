
import streamlit as st
import requests
import pandas as pd
import plotly.express as px


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="LLM-EvalHub",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 LLM-EvalHub")
st.caption("Automated LLM Evaluation & Analytics Platform")


# ============================================================
# BACKEND CONFIGURATION
# ============================================================

BACKEND_URL = "http://127.0.0.1:8001"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_json(response):
    try:
        return response.json()
    except Exception:
        return {}


def get_datasets():
    try:
        response = requests.get(
            f"{BACKEND_URL}/datasets",
            timeout=10,
        )

        if response.status_code == 200:
            data = response.json()

            if isinstance(data, list):
                return data

            return []

        st.error(
            f"Backend error while loading datasets: "
            f"{response.status_code}"
        )

        return []

    except requests.exceptions.RequestException:
        st.error(
            "Cannot connect to FastAPI backend. "
            "Make sure the backend is running on port 8001."
        )

        return []


def get_prompts():
    try:
        response = requests.get(
            f"{BACKEND_URL}/prompts",
            timeout=10,
        )

        if response.status_code == 200:
            data = response.json()

            if isinstance(data, list):
                return data

            return []

        return []

    except requests.exceptions.RequestException:
        return []


def get_run_comparison(run_ids):
    try:
        params = []

        for run_id in run_ids:
            params.append(("run_ids", int(run_id)))

        response = requests.get(
            f"{BACKEND_URL}/compare",
            params=params,
            timeout=30,
        )

        return response

    except requests.exceptions.RequestException as exc:
        st.error(
            f"Backend connection failed: {exc}"
        )

        return None


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📚 Create Dataset",
        "🚀 Run Evaluation",
        "📊 Results",
        "⚖️ Compare Runs",
    ]
)


# ============================================================
# TAB 1 — CREATE DATASET
# ============================================================

with tab1:

    st.header("📚 Create Evaluation Dataset")

    dataset_name = st.text_input(
        "Dataset name",
        placeholder="Example: LLM Evaluation QA v3",
    )

    st.subheader("Question 1")

    q1 = st.text_area(
        "Question",
        key="q1",
        placeholder="Enter evaluation question",
    )

    c1 = st.text_area(
        "Context",
        key="c1",
        placeholder="Optional context",
    )

    e1 = st.text_area(
        "Expected answer",
        key="e1",
        placeholder="Expected answer",
    )

    category1 = st.text_input(
        "Category",
        value="General",
        key="category1",
    )

    difficulty1 = st.selectbox(
        "Difficulty",
        [
            "Easy",
            "Medium",
            "Hard",
        ],
        key="difficulty1",
    )

    if st.button(
        "Create Dataset",
        type="primary",
    ):

        if not dataset_name or not q1:

            st.warning(
                "Please enter at least the dataset name and question."
            )

        else:

            dataset_payload = {
                "name": dataset_name,
                "prompts": [
                    {
                        "question": q1,
                        "context": c1,
                        "expected_output": e1,
                        "category": category1,
                        "difficulty": difficulty1,
                    }
                ],
            }

            try:

                response = requests.post(
                    f"{BACKEND_URL}/datasets",
                    json=dataset_payload,
                    timeout=30,
                )

                if response.status_code in [200, 201]:

                    st.success(
                        "Dataset created successfully!"
                    )

                    st.json(
                        safe_json(response)
                    )

                else:

                    st.error(
                        f"Dataset creation failed: "
                        f"{response.status_code}"
                    )

                    st.code(
                        response.text
                    )

            except requests.exceptions.RequestException as exc:

                st.error(
                    f"Backend connection failed: {exc}"
                )


# ============================================================
# TAB 2 — RUN EVALUATION
# ============================================================

with tab2:

    st.header("🚀 Run LLM Evaluation")

    datasets = get_datasets()

    if datasets:

        dataset_labels = {
            f"{d['id']} — {d['name']}": d["id"]
            for d in datasets
        }

        selected_dataset_label = st.selectbox(
            "Select Dataset",
            list(dataset_labels.keys()),
        )

        dataset_id = dataset_labels[
            selected_dataset_label
        ]

    else:

        st.warning(
            "No datasets available."
        )

        dataset_id = None

    st.subheader("🤖 Model Selection")

    model_options = [
        "mock-model",
        "gpt-4o-mini",
        "gemini-3.6-flash",
        "mistral-small-latest",
        "codestral-2508",
    ]

    selected_models = st.multiselect(
        "Select model(s) to evaluate",
        model_options,
        default=["codestral-2508"],
    )

    st.subheader("📝 Prompt Configuration")

    prompt_version = st.selectbox(
        "Prompt version",
        [
            "v1",
            "v2",
        ],
        index=1,
    )

    if prompt_version == "v1":

        st.info(
            "v1 — Answer the question using the provided context."
        )

    else:

        st.info(
            "v2 — Answer using only the provided context. "
            "If the answer is not supported, state that the "
            "information is insufficient."
        )

    st.subheader("🧪 Experiment Configuration")

    run_name = st.text_input(
        "Experiment name",
        value="experiment-codestral-2508",
    )

    temperature = st.number_input(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.0,
        step=0.1,
    )

    max_tokens = st.number_input(
        "Maximum output tokens",
        min_value=1,
        max_value=4096,
        value=256,
        step=1,
    )

    st.info(
        "The backend records model responses, latency, "
        "token usage, cost, standard metrics, and "
        "LLM-as-a-Judge metrics."
    )

    if st.button(
        "▶️ Run Evaluation",
        type="primary",
    ):

        if not dataset_id:

            st.error(
                "Please select a dataset."
            )

        elif not selected_models:

            st.error(
                "Please select at least one model."
            )

        else:

            run_payload = {
                "name": run_name,
                "dataset_id": dataset_id,
                "models": selected_models,
                "prompt_version": prompt_version,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            with st.spinner(
                "Running LLM evaluation..."
            ):

                try:

                    response = requests.post(
                        f"{BACKEND_URL}/runs",
                        json=run_payload,
                        timeout=300,
                    )

                    if response.status_code in [
                        200,
                        201,
                    ]:

                        result = safe_json(response)

                        status = result.get(
                            "status",
                            "completed",
                        )

                        run_id = result.get(
                            "run_id"
                        )

                        # ------------------------------------------------
                        # SUCCESS / PARTIAL SUCCESS
                        # ------------------------------------------------

                        if status == "completed_with_errors":

                            st.warning(
                                "⚠️ Evaluation completed with one or more "
                                "model failures."
                            )

                        elif status == "completed":

                            st.success(
                                "🎉 Evaluation completed successfully!"
                            )

                        else:

                            st.info(
                                f"Evaluation status: {status}"
                            )

                        # ------------------------------------------------
                        # RUN SUMMARY
                        # ------------------------------------------------

                        st.subheader("Run Summary")

                        summary_col1, summary_col2, summary_col3 = st.columns(3)

                        with summary_col1:

                            st.metric(
                                "Run ID",
                                run_id if run_id is not None else "N/A",
                            )

                        with summary_col2:

                            st.metric(
                                "Responses",
                                result.get(
                                    "response_count",
                                    0,
                                ),
                            )

                        with summary_col3:

                            st.metric(
                                "Judge Type",
                                result.get(
                                    "judge_type",
                                    "N/A",
                                ),
                            )

                        # ------------------------------------------------
                        # MODEL STATUS
                        # ------------------------------------------------

                        successful_models = result.get(
                            "successful_models",
                            [],
                        )

                        failed_models = result.get(
                            "failed_models",
                            [],
                        )

                        if successful_models:

                            st.success(
                                "✅ Successful models: "
                                + ", ".join(successful_models)
                            )

                        if failed_models:

                            st.error(
                                f"❌ Failed models: "
                                f"{len(failed_models)}"
                            )

                            failure_rows = []

                            for failure in failed_models:

                                failure_rows.append(
                                    {
                                        "Model": failure.get(
                                            "model_name",
                                            "Unknown",
                                        ),
                                        "Question": failure.get(
                                            "question",
                                            "",
                                        ),
                                        "Error Type": failure.get(
                                            "error_type",
                                            "",
                                        ),
                                        "Error": failure.get(
                                            "error",
                                            "",
                                        ),
                                    }
                                )

                            if failure_rows:

                                st.dataframe(
                                    pd.DataFrame(
                                        failure_rows
                                    ),
                                    use_container_width=True,
                                )

                        # ------------------------------------------------
                        # EVALUATION COUNTS
                        # ------------------------------------------------

                        count_col1, count_col2, count_col3, count_col4 = st.columns(4)

                        with count_col1:

                            st.metric(
                                "Standard Scores",
                                result.get(
                                    "standard_score_count",
                                    0,
                                ),
                            )

                        with count_col2:

                            st.metric(
                                "Judge Scores",
                                result.get(
                                    "judge_score_count",
                                    0,
                                ),
                            )

                        with count_col3:

                            st.metric(
                                "Real LLM Judges",
                                result.get(
                                    "llm_judge_count",
                                    0,
                                ),
                            )

                        with count_col4:

                            st.metric(
                                "Local Fallbacks",
                                result.get(
                                    "local_fallback_count",
                                    0,
                                ),
                            )

                        # ------------------------------------------------
                        # STORE LATEST RUN
                        # ------------------------------------------------

                        if run_id is not None:

                            st.session_state[
                                "latest_run_id"
                            ] = int(run_id)

                            st.info(
                                f"Latest Run ID: {run_id}"
                            )

                        # ------------------------------------------------
                        # FULL BACKEND RESPONSE
                        # ------------------------------------------------

                        with st.expander(
                            "View complete backend response"
                        ):

                            st.json(result)

                    else:

                        st.error(
                            f"Evaluation failed: "
                            f"{response.status_code}"
                        )

                        st.code(
                            response.text
                        )

                except requests.exceptions.RequestException as exc:

                    st.error(
                        f"Backend connection failed: {exc}"
                    )


# ============================================================
# TAB 3 — RESULTS
# ============================================================

with tab3:

    st.header("📊 Evaluation Results")

    default_run_id = st.session_state.get(
        "latest_run_id",
        27,
    )

    run_id = st.number_input(
        "Run ID",
        min_value=1,
        value=int(default_run_id),
        step=1,
    )

    if st.button(
        "🔄 Load Results",
        type="primary",
    ):

        # ====================================================
        # STANDARD RESULTS
        # ====================================================

        try:

            results_response = requests.get(
                f"{BACKEND_URL}/runs/{run_id}/results",
                timeout=30,
            )

            if results_response.status_code == 200:

                results = results_response.json()

                st.subheader(
                    "📈 Standard Evaluation Metrics"
                )

                if results:

                    rows = []

                    for response_item in results:

                        for score in response_item.get(
                            "scores",
                            [],
                        ):

                            rows.append(
                                {
                                    "Response ID": response_item.get(
                                        "id"
                                    ),
                                    "Model": response_item.get(
                                        "model_name"
                                    ),
                                    "Metric": score.get(
                                        "metric_name"
                                    ),
                                    "Score": score.get(
                                        "score_value"
                                    ),
                                }
                            )

                    if rows:

                        df = pd.DataFrame(rows)

                        st.dataframe(
                            df,
                            use_container_width=True,
                        )

                        metric_summary = (
                            df.groupby(
                                "Metric",
                                as_index=False,
                            )["Score"]
                            .mean()
                        )

                        fig = px.bar(
                            metric_summary,
                            x="Metric",
                            y="Score",
                            title="Average Standard Metrics",
                            range_y=[0, 1],
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                        )

                    else:

                        st.info(
                            "No standard scores found."
                        )

                else:

                    st.info(
                        "No responses found for this run."
                    )

            else:

                st.error(
                    f"Could not load standard results: "
                    f"{results_response.status_code}"
                )

                st.code(
                    results_response.text
                )

        except requests.exceptions.RequestException as exc:

            st.error(
                f"Backend connection failed: {exc}"
            )

        # ====================================================
        # LLM-AS-A-JUDGE RESULTS
        # ====================================================

        st.divider()

        st.subheader(
            "🧑‍⚖️ LLM-as-a-Judge Results"
        )

        try:

            judge_response = requests.get(
                f"{BACKEND_URL}/runs/{run_id}/judge-results",
                timeout=30,
            )

            if judge_response.status_code == 200:

                judge_results = judge_response.json()

                if isinstance(
                    judge_results,
                    dict,
                ):

                    judge_items = judge_results.get(
                        "results",
                        [],
                    )

                elif isinstance(
                    judge_results,
                    list,
                ):

                    judge_items = judge_results

                else:

                    judge_items = []

                if judge_items:

                    judge_rows = []

                    for item in judge_items:

                        if not isinstance(
                            item,
                            dict,
                        ):
                            continue

                        response_id = item.get(
                            "response_id"
                        )

                        model_name = item.get(
                            "model_name",
                            "",
                        )

                        judge_scores = item.get(
                            "judge_scores",
                            [],
                        )

                        if not isinstance(
                            judge_scores,
                            list,
                        ):
                            continue

                        for score in judge_scores:

                            if not isinstance(
                                score,
                                dict,
                            ):
                                continue

                            judge_rows.append(
                                {
                                    "Response ID": response_id,
                                    "Model": model_name,
                                    "Judge Type": score.get(
                                        "judge_type"
                                    ),
                                    "Rubric": score.get(
                                        "rubric_version"
                                    ),
                                    "Metric": score.get(
                                        "metric_name"
                                    ),
                                    "Score": score.get(
                                        "score_value"
                                    ),
                                    "Explanation": score.get(
                                        "explanation"
                                    ),
                                }
                            )

                    if judge_rows:

                        judge_df = pd.DataFrame(
                            judge_rows
                        )

                        st.dataframe(
                            judge_df,
                            use_container_width=True,
                        )

                        judge_summary = (
                            judge_df.groupby(
                                "Metric",
                                as_index=False,
                            )["Score"]
                            .mean()
                        )

                        fig_judge = px.bar(
                            judge_summary,
                            x="Metric",
                            y="Score",
                            title="Average LLM-as-a-Judge Metrics",
                            range_y=[0, 1],
                        )

                        st.plotly_chart(
                            fig_judge,
                            use_container_width=True,
                        )

                        # ====================================
                        # JUDGE TYPE SUMMARY
                        # ====================================

                        st.subheader(
                            "🔍 Judge Execution Summary"
                        )

                        judge_types = (
                            judge_df[
                                "Judge Type"
                            ]
                            .value_counts()
                            .reset_index()
                        )

                        judge_types.columns = [
                            "Judge Type",
                            "Count",
                        ]

                        st.dataframe(
                            judge_types,
                            use_container_width=True,
                        )

                        # ====================================
                        # HALLUCINATION ANALYSIS
                        # ====================================

                        st.subheader(
                            "🚨 Hallucination Analysis"
                        )

                        hallucination_df = judge_df[
                            judge_df["Metric"]
                            == "hallucination"
                        ]

                        if not hallucination_df.empty:

                            st.dataframe(
                                hallucination_df,
                                use_container_width=True,
                            )

                            fig_hallucination = px.bar(
                                hallucination_df,
                                x="Response ID",
                                y="Score",
                                title="Hallucination Score by Response",
                                range_y=[0, 1],
                            )

                            st.plotly_chart(
                                fig_hallucination,
                                use_container_width=True,
                            )

                        else:

                            st.info(
                                "No hallucination scores found."
                            )

                        # ====================================
                        # RESPONSE-LEVEL DETAILS
                        # ====================================

                        st.subheader(
                            "🔎 Response-Level Judge Details"
                        )

                        for response_id in sorted(
                            judge_df["Response ID"]
                            .dropna()
                            .unique()
                        ):

                            response_data = judge_df[
                                judge_df["Response ID"]
                                == response_id
                            ]

                            with st.expander(
                                f"Response {response_id}"
                            ):

                                st.write(
                                    "Model:",
                                    response_data[
                                        "Model"
                                    ].iloc[0],
                                )

                                st.write(
                                    "Judge Type:",
                                    response_data[
                                        "Judge Type"
                                    ].iloc[0],
                                )

                                st.write(
                                    "Rubric:",
                                    response_data[
                                        "Rubric"
                                    ].iloc[0],
                                )

                                display_columns = [
                                    "Metric",
                                    "Score",
                                    "Explanation",
                                ]

                                st.dataframe(
                                    response_data[
                                        display_columns
                                    ],
                                    use_container_width=True,
                                )

                    else:

                        st.info(
                            "No judge scores found for this run."
                        )

                else:

                    st.info(
                        "No judge results found for this run."
                    )

            else:

                st.error(
                    f"Could not load judge results: "
                    f"{judge_response.status_code}"
                )

                st.code(
                    judge_response.text
                )

        except requests.exceptions.RequestException as exc:

            st.error(
                f"Backend connection failed: {exc}"
            )


# ============================================================
# TAB 4 — COMPARE RUNS
# ============================================================

with tab4:

    st.header("⚖️ Compare Evaluation Runs")

    st.write(
        "Compare two or more completed runs using their "
        "aggregated model metrics."
    )

    comparison_run_1 = st.number_input(
        "Run ID 1",
        min_value=1,
        value=25,
        step=1,
        key="comparison_run_1",
    )

    comparison_run_2 = st.number_input(
        "Run ID 2",
        min_value=1,
        value=27,
        step=1,
        key="comparison_run_2",
    )

    if st.button(
        "⚖️ Compare Runs",
        type="primary",
    ):

        if comparison_run_1 == comparison_run_2:

            st.warning(
                "Please select two different run IDs."
            )

        else:

            response = get_run_comparison(
                [
                    comparison_run_1,
                    comparison_run_2,
                ]
            )

            if response is not None:

                if response.status_code == 200:

                    comparison = response.json()

                    st.success(
                        "Comparison loaded successfully."
                    )

                    # --------------------------------------------
                    # RUN INFORMATION
                    # --------------------------------------------

                    st.subheader(
                        "Experiment Information"
                    )

                    comparison_runs = comparison.get(
                        "runs",
                        [],
                    )

                    run_rows = []

                    for run in comparison_runs:

                        run_rows.append(
                            {
                                "Run ID": run.get(
                                    "run_id"
                                ),
                                "Run Name": run.get(
                                    "run_name"
                                ),
                                "Dataset ID": run.get(
                                    "dataset_id"
                                ),
                                "Prompt Version": run.get(
                                    "prompt_version"
                                ),
                                "Temperature": run.get(
                                    "temperature"
                                ),
                                "Max Tokens": run.get(
                                    "max_tokens"
                                ),
                            }
                        )

                    if run_rows:

                        st.dataframe(
                            pd.DataFrame(run_rows),
                            use_container_width=True,
                        )

                    # --------------------------------------------
                    # MODEL METRICS
                    # --------------------------------------------

                    comparison_rows = []

                    for run in comparison_runs:

                        for model in run.get(
                            "models",
                            [],
                        ):

                            standard_metrics = model.get(
                                "standard_metrics",
                                {},
                            )

                            judge_metrics = model.get(
                                "judge_metrics",
                                {},
                            )

                            comparison_rows.append(
                                {
                                    "Run ID": run.get(
                                        "run_id"
                                    ),
                                    "Model": model.get(
                                        "model_name"
                                    ),
                                    "Responses": model.get(
                                        "response_count"
                                    ),
                                    "Avg Latency (ms)": model.get(
                                        "average_latency_ms"
                                    ),
                                    "Total Cost (USD)": model.get(
                                        "total_cost_usd"
                                    ),
                                    "Token F1": standard_metrics.get(
                                        "token_f1"
                                    ),
                                    "TF-IDF Similarity": standard_metrics.get(
                                        "tfidf_cosine_similarity"
                                    ),
                                    "Correctness": judge_metrics.get(
                                        "correctness"
                                    ),
                                    "Relevance": judge_metrics.get(
                                        "relevance"
                                    ),
                                    "Faithfulness": judge_metrics.get(
                                        "faithfulness"
                                    ),
                                    "Coherence": judge_metrics.get(
                                        "coherence"
                                    ),
                                    "Hallucination": judge_metrics.get(
                                        "hallucination"
                                    ),
                                    "Judge Type": model.get(
                                        "judge_type"
                                    ),
                                }
                            )

                    if comparison_rows:

                        comparison_df = pd.DataFrame(
                            comparison_rows
                        )

                        st.subheader(
                            "📊 Model Comparison"
                        )

                        st.dataframe(
                            comparison_df,
                            use_container_width=True,
                        )

                        # ----------------------------------------
                        # LATENCY CHART
                        # ----------------------------------------

                        latency_df = comparison_df[
                            [
                                "Run ID",
                                "Model",
                                "Avg Latency (ms)",
                            ]
                        ].copy()

                        fig_latency = px.bar(
                            latency_df,
                            x="Model",
                            y="Avg Latency (ms)",
                            color="Run ID",
                            title="Average Latency",
                        )

                        st.plotly_chart(
                            fig_latency,
                            use_container_width=True,
                        )

                        # ----------------------------------------
                        # COST CHART
                        # ----------------------------------------

                        cost_df = comparison_df[
                            [
                                "Run ID",
                                "Model",
                                "Total Cost (USD)",
                            ]
                        ].copy()

                        fig_cost = px.bar(
                            cost_df,
                            x="Model",
                            y="Total Cost (USD)",
                            color="Run ID",
                            title="Total Evaluation Cost",
                        )

                        st.plotly_chart(
                            fig_cost,
                            use_container_width=True,
                        )

                        # ----------------------------------------
                        # JUDGE METRICS
                        # ----------------------------------------

                        judge_columns = [
                            "Run ID",
                            "Model",
                            "Correctness",
                            "Relevance",
                            "Faithfulness",
                            "Coherence",
                            "Hallucination",
                        ]

                        judge_comparison_df = comparison_df[
                            judge_columns
                        ].copy()

                        judge_long_df = judge_comparison_df.melt(
                            id_vars=[
                                "Run ID",
                                "Model",
                            ],
                            value_vars=[
                                "Correctness",
                                "Relevance",
                                "Faithfulness",
                                "Coherence",
                                "Hallucination",
                            ],
                            var_name="Metric",
                            value_name="Score",
                        )

                        fig_judge_compare = px.bar(
                            judge_long_df,
                            x="Metric",
                            y="Score",
                            color="Model",
                            barmode="group",
                            title="LLM-as-a-Judge Comparison",
                            range_y=[0, 1],
                        )

                        st.plotly_chart(
                            fig_judge_compare,
                            use_container_width=True,
                        )

                    else:

                        st.info(
                            "No model comparison data was returned."
                        )

                else:

                    st.error(
                        f"Comparison failed: "
                        f"{response.status_code}"
                    )

                    st.code(
                        response.text
                    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "LLM-EvalHub • Automated LLM Evaluation & Analytics Platform"
)