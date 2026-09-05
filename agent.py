import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from xgboost import XGBRegressor

from visualizations import get_pressure_trend_figure, get_shap_contributions_figure

TARGET = "reactor_pressure"
TAG_COLUMNS = ["fault_number", "split", "run_id", "sample_index"]
FAULT_ONSET_SAMPLE = 160
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768
RAG_TOP_K = 3

PLOTS_DIR = None  # set in main(), based on data_dir

# ---------------------------------------------------------------------------
# Load data + train twin + load RAG corpus once at startup. The tool
# functions below close over these variables so Gemini can call them with
# just the arguments it needs.
# ---------------------------------------------------------------------------

TRAIN_DF = None
TEST_DF = None
FEATURE_COLS = None
TWIN_MODEL = None
EXPLAINER = None
BASELINE_MEAN_SHAP = None

RAG_EMBEDDINGS = None   # numpy array, shape (n_chunks, EMBEDDING_DIM)
RAG_METADATA = None     # list of dicts: {"source", "chunk_index", "text"}
GENAI_CLIENT = None     # needed inside retrieve_manual to embed the query

# Maps raw filenames to proper citations, so the agent references sources
# naturally instead of repeating back "TEP_pdf2.pdf".
SOURCE_DISPLAY_NAMES = {
    "TEP_pdf.pdf": "Yapur (2022), \"Advantages of OKID-ERA Identification in Control Systems\"",
    "TEP_pdf2.pdf": "Duvall (1996), \"On-Line Optimization of the Tennessee Eastman Challenge Problem\" (Texas Tech thesis)",
    "tep_plant_overview.md": "TEP Plant Overview (project reference document)",
}


def _load_and_train(data_dir: Path):
    global TRAIN_DF, TEST_DF, FEATURE_COLS, TWIN_MODEL, EXPLAINER, BASELINE_MEAN_SHAP

    TRAIN_DF = pd.read_parquet(data_dir / "tep_train_renamed.parquet")
    TEST_DF = pd.read_parquet(data_dir / "tep_test_renamed.parquet")
    FEATURE_COLS = [c for c in TRAIN_DF.columns if c not in TAG_COLUMNS + [TARGET]]

    normal_train = TRAIN_DF[TRAIN_DF["fault_number"] == 0]
    X_train = normal_train[FEATURE_COLS]
    y_train = normal_train[TARGET]

    TWIN_MODEL = XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
    )
    TWIN_MODEL.fit(X_train, y_train)

    EXPLAINER = shap.TreeExplainer(TWIN_MODEL)

    normal_test = TEST_DF[TEST_DF["fault_number"] == 0]
    shap_normal = EXPLAINER.shap_values(normal_test[FEATURE_COLS])
    BASELINE_MEAN_SHAP = np.mean(shap_normal, axis=0)

    print(f"Twin trained and ready. ({len(normal_train)} normal training rows)")


def _load_rag_corpus(data_dir: Path):
    global RAG_EMBEDDINGS, RAG_METADATA

    embeddings_path = data_dir / "rag_embeddings.npy"
    metadata_path = data_dir / "rag_metadata.json"

    if not embeddings_path.exists() or not metadata_path.exists():
        print("Warning: RAG corpus files not found. retrieve_manual will be unavailable "
              "until you run ingest_rag_corpus.py.")
        return

    RAG_EMBEDDINGS = np.load(embeddings_path)
    with open(metadata_path) as f:
        RAG_METADATA = json.load(f)

    print(f"RAG corpus loaded: {len(RAG_METADATA)} chunks from "
          f"{len(set(c['source'] for c in RAG_METADATA))} documents.")


# ---------------------------------------------------------------------------
# Tool functions -- Gemini calls these directly. Docstrings are what the
# model reads to decide when and how to call each one, so keep them precise.
# ---------------------------------------------------------------------------

def run_digital_twin(fault_number: int) -> dict:
    """Runs the digital twin on a given TEP fault number and reports how far
    actual reactor pressure deviates from the twin's predicted (expected/healthy)
    reactor pressure, using post-fault-onset test data.

    Args:
        fault_number: The TEP fault number to analyze (0 = normal operation, 1-21 = fault types).

    Returns:
        A dict with mean actual pressure, mean predicted pressure, mean residual,
        and a qualitative severity label.
    """
    onset_filter = TEST_DF["sample_index"] >= (FAULT_ONSET_SAMPLE if fault_number != 0 else 0)
    subset = TEST_DF[(TEST_DF["fault_number"] == fault_number) & onset_filter]
    if subset.empty:
        return {"error": f"No data found for fault_number={fault_number}"}

    X = subset[FEATURE_COLS]
    actual = subset[TARGET].values
    predicted = TWIN_MODEL.predict(X)
    residual = actual - predicted

    mean_residual = float(np.mean(residual))
    abs_residual = abs(mean_residual)
    if abs_residual < 5:
        severity = "negligible"
    elif abs_residual < 50:
        severity = "moderate"
    else:
        severity = "severe"

    return {
        "fault_number": fault_number,
        "mean_actual_pressure": round(float(np.mean(actual)), 2),
        "mean_predicted_pressure": round(float(np.mean(predicted)), 2),
        "mean_residual": round(mean_residual, 4),
        "severity": severity,
    }


def explain_root_cause(fault_number: int) -> dict:
    """Runs SHAP analysis to identify which process variables are the most
    likely root cause of reactor pressure deviation for a given TEP fault,
    by comparing SHAP contributions against normal-operation baseline.

    Args:
        fault_number: The TEP fault number to analyze (1-21; use run_digital_twin
            first to check whether a fault produces a meaningful deviation at all).

    Returns:
        A dict with the top contributing variables and their SHAP contribution
        shift relative to normal baseline (larger magnitude = stronger root-cause signal).
    """
    subset = TEST_DF[
        (TEST_DF["fault_number"] == fault_number)
        & (TEST_DF["sample_index"] >= FAULT_ONSET_SAMPLE)
    ]
    if subset.empty:
        return {"error": f"No post-onset data found for fault_number={fault_number}"}

    X = subset[FEATURE_COLS]
    shap_values = EXPLAINER.shap_values(X)
    fault_mean_shap = np.mean(shap_values, axis=0)
    shift = fault_mean_shap - BASELINE_MEAN_SHAP

    shift_series = pd.Series(shift, index=FEATURE_COLS).sort_values(key=abs, ascending=False)
    top_5 = shift_series.head(5)

    return {
        "fault_number": fault_number,
        "top_contributing_variables": {k: round(float(v), 4) for k, v in top_5.items()},
    }


def plot_pressure_trend(fault_number: int) -> dict:
    """Generates a chart comparing actual vs. digital-twin-predicted reactor
    pressure over time for a given fault, with the fault onset point marked.
    Saves the chart as a PNG file.

    Args:
        fault_number: The TEP fault number to plot (0 = normal, 1-21 = fault types).

    Returns:
        A dict with the saved file path and a short description of what the chart shows.
    """
    try:
        fig = get_pressure_trend_figure(TEST_DF, TWIN_MODEL, FEATURE_COLS, TARGET, fault_number)
    except ValueError as e:
        return {"error": str(e)}

    out_path = PLOTS_DIR / f"pressure_trend_fault_{fault_number}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return {
        "fault_number": fault_number,
        "chart_saved_to": str(out_path),
        "description": f"Line chart of actual vs. predicted reactor pressure over time for fault {fault_number}.",
    }


def plot_shap_contributions(fault_number: int) -> dict:
    """Generates a bar chart of the top root-cause variables (from SHAP
    analysis) for a given fault, ranked by contribution shift vs. normal
    baseline. Saves the chart as a PNG file.

    Note: call explain_root_cause first to confirm the fault has a
    meaningful deviation before generating this chart.

    Args:
        fault_number: The TEP fault number to plot (1-21).

    Returns:
        A dict with the saved file path and a short description of what the chart shows.
    """
    subset = TEST_DF[
        (TEST_DF["fault_number"] == fault_number)
        & (TEST_DF["sample_index"] >= FAULT_ONSET_SAMPLE)
    ]
    if subset.empty:
        return {"error": f"No post-onset data found for fault_number={fault_number}"}

    X = subset[FEATURE_COLS]
    shap_values = EXPLAINER.shap_values(X)
    fault_mean_shap = np.mean(shap_values, axis=0)
    shift = fault_mean_shap - BASELINE_MEAN_SHAP
    shift_series = pd.Series(shift, index=FEATURE_COLS).sort_values(key=abs, ascending=False)

    fig = get_shap_contributions_figure(shift_series, fault_number)
    out_path = PLOTS_DIR / f"shap_contributions_fault_{fault_number}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return {
        "fault_number": fault_number,
        "chart_saved_to": str(out_path),
        "description": f"Bar chart of top root-cause variables for fault {fault_number}.",
    }


def _cosine_similarity(query_vec: np.ndarray, corpus_matrix: np.ndarray) -> np.ndarray:
    query_norm = query_vec / np.linalg.norm(query_vec)
    corpus_norms = corpus_matrix / np.linalg.norm(corpus_matrix, axis=1, keepdims=True)
    return corpus_norms @ query_norm


def retrieve_manual(query: str) -> dict:
    """Searches the plant process documentation (technical papers and a plain-
    language plant overview) for passages relevant to a question about how the
    plant works, what a piece of equipment does, or what a fault means
    physically. Use this for conceptual/explanatory questions about the
    process itself -- not for numeric fault analysis, which run_digital_twin
    and explain_root_cause handle.

    Args:
        query: The question or topic to search the documentation for.

    Returns:
        A dict with the top matching passages, each including its source
        document and text.
    """
    if RAG_EMBEDDINGS is None or RAG_METADATA is None:
        return {"error": "RAG corpus not loaded. Run ingest_rag_corpus.py first."}

    from google.genai import types

    result = GENAI_CLIENT.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=EMBEDDING_DIM,
        ),
    )
    query_vec = np.array(result.embeddings[0].values)

    similarities = _cosine_similarity(query_vec, RAG_EMBEDDINGS)
    top_indices = np.argsort(similarities)[::-1][:RAG_TOP_K]

    matches = []
    for idx in top_indices:
        chunk = RAG_METADATA[idx]
        source_label = SOURCE_DISPLAY_NAMES.get(chunk["source"], chunk["source"])
        matches.append({
            "source": source_label,
            "similarity": round(float(similarities[idx]), 4),
            "text": chunk["text"],
        })

    return {"query": query, "matches": matches}


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTION = """
You are an experienced process engineer analyzing a chemical plant (the Tennessee
Eastman Process benchmark). You have access to five tools:
- run_digital_twin: reports how much reactor pressure deviates from expected/healthy
  behavior for a given fault.
- explain_root_cause: uses SHAP analysis to identify which process variables are
  driving that deviation.
- plot_pressure_trend: generates a chart of actual vs. predicted pressure over time.
- plot_shap_contributions: generates a bar chart of the top root-cause variables.
- retrieve_manual: searches plant process documentation for conceptual/explanatory
  answers about how the plant, its equipment, or a fault type works.

Decide which tools a question needs:
- Numeric fault analysis ("analyze fault 6", "what's the deviation") -> run_digital_twin,
  then explain_root_cause if the deviation is not negligible.
- Charts ("show me", "plot", "visualize") -> plot_pressure_trend / plot_shap_contributions,
  only when explicitly asked.
- Conceptual/explanatory questions ("what does the stripper do", "why would cooling
  water faults be less severe", "what is fault 6 physically") -> retrieve_manual.
  Use the retrieved passages to inform your answer, but answer naturally in your own
  words, the way an experienced engineer would from memory -- do not mention that you
  searched documentation, do not cite filenames or paper titles, and do not say
  phrases like "based on the documentation" or "according to my sources."
- Some questions warrant both: e.g. "analyze fault 6 and explain what that means for
  the plant" should call run_digital_twin/explain_root_cause AND retrieve_manual.

Write your findings the way a process engineer would in a shift report or handover
note -- plain, direct, grounded in the numbers or documentation, no hedging filler.
If a numeric deviation is negligible, say so plainly and explain that the fault may
be well-compensated by existing control loops rather than assuming something is
wrong with the analysis. When you generate a chart, mention the file path so the
user can open it. Never mention tool names, function calls, or that you retrieved
information from a document -- just state what you know, the way a colleague would.

Do not speculate beyond what the tool outputs and retrieved documentation show you.
"""


def main():
    if len(sys.argv) != 2:
        print("Usage: python agent.py /path/to/data/folder")
        sys.exit(1)

    if not os.environ.get("GOOGLE_API_KEY"):
        print("Error: set your GOOGLE_API_KEY environment variable first.")
        print('  export GOOGLE_API_KEY="your_key_here"')
        sys.exit(1)

    from google import genai
    from google.genai import types

    data_dir = Path(sys.argv[1])

    global PLOTS_DIR, GENAI_CLIENT
    PLOTS_DIR = data_dir / "plots"
    PLOTS_DIR.mkdir(exist_ok=True)

    _load_and_train(data_dir)
    _load_rag_corpus(data_dir)

    GENAI_CLIENT = genai.Client()
    config = types.GenerateContentConfig(
        tools=[
            run_digital_twin,
            explain_root_cause,
            plot_pressure_trend,
            plot_shap_contributions,
            retrieve_manual,
        ],
        system_instruction=SYSTEM_INSTRUCTION,
    )

    print("\nTEP Digital Twin Agent ready. Ask about any fault (0-21), or type 'quit'.\n")

    chat = GENAI_CLIENT.chats.create(model="gemini-3.1-pro-preview", config=config)

    while True:
        user_input = input("> ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        if not user_input:
            continue

        response = chat.send_message(user_input)
        print(f"\n{response.text}\n")


if __name__ == "__main__":
    main()
