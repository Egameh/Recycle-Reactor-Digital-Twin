# Recycle-Reactor Digital Twin

A digital twin + LLM agent for the Tennessee Eastman Process (TEP) benchmark.
Combining a statistical process model, SHAP-based root-cause analysis, and a
tool-calling Gemini agent grounded in real process documentation (RAG).

Built as a DEMO project bridging industrial process engineering and
applied AI/ML: reliability-focused digital twin deployment, explainability,
and agentic reasoning over a real (simulated) chemical plant.

## What it does

As a Process engineer, you can ask the agent about a plant fault, and it:

1. Runs a trained digital twin to check whether the fault causes a
   meaningful deviation in reactor pressure
2. If it does, runs a SHAP analysis to identify which process variables are
   the likely root cause
3. Can generate charts of the deviation and the root-cause ranking
4. Can answer conceptual questions about the plant itself, grounded in
   real process engineering documentation, via retrieval-augmented generation

All synthesized into plain, direct engineer-style commentary

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Gemini Agent                        │
│         (decides which tools a question needs)        │
└───────────────────────┬───────────────────────────────┘
                         │
      ┌──────────────────┼──────────────────┬─────────────────┐
      ▼                  ▼                  ▼                 ▼
┌───────────┐    ┌───────────────┐   ┌─────────────┐   ┌──────────────┐
│  Digital   │    │  SHAP Root-   │   │ Visualization│   │  RAG Layer   │
│   Twin     │───▶│  Cause        │   │   Tools      │   │ (retrieval   │
│ (XGBoost   │    │  Explainer    │   │ (matplotlib) │   │ over process │
│ regressor) │    │               │   │              │   │  docs)       │
└───────────┘    └───────────────┘   └─────────────┘   └──────────────┘
```

**Digital twin**: an XGBoost regressor trained on normal-operation data
only, predicting reactor pressure from the plant's other 51 process
variables. Deviations between actual and predicted pressure (residuals)
are the anomaly signal.

**Root-cause explainer**: rather than a fixed fault classifier, SHAP values
compare each fault's variable contributions against the normal-operation
baseline. This generalizes to explaining *why* a deviation is happening
without needing the fault to match a predefined category. Closer to how
a process engineer actually reasons about an anomaly.

**RAG layer**: process engineering documentation (a Texas Tech thesis on
TEP optimization, a paper on TEP system identification, and a plain-
language plant overview) is chunked and embedded with Gemini's embedding
model. Conceptual questions ("what does the stripper do", "why is this
plant hard to control") are answered by retrieving and synthesizing
relevant passages — without the agent narrating that it searched anything.

## Example results

**Fault 4 (Reactor Cooling Water Inlet Temperature Step):**
Mean residual: **+0.63** — negligible. The plant's control loops absorb
this disturbance almost completely before it reaches reactor pressure.

**Fault 6 (A Feed Loss):**
Mean residual: **+260.94** — severe. SHAP correctly surfaces
`product_sep_pressure`, `stripper_pressure`, and feed-A-related variables
as the top contributors, without ever being told what fault 6 actually is.

This contrast is the core validation of the approach: subtle,
well-compensated faults and severe, poorly-compensated faults produce
clearly different, physically coherent diagnostic signatures.

## Setup

```bash
pip install -r requirements.txt
export GOOGLE_API_KEY="your_key_here"
```

### 1. Get the data

Download the Tennessee Eastman Process dataset (Downs & Vogel / Rieth et
al. simulation format — `d00.dat` through `d21_te.dat`) and place all 44
files in a folder, e.g. `TEP_data/`.

### 2. Build the pipeline

```bash
python tep_loader.py /path/to/TEP_data
python rename_tep_columns.py /path/to/TEP_data
```

### 3. (Optional) Build the RAG corpus

Gather your own process documentation (PDFs or Markdown), then:

```bash
python ingest_rag_corpus.py /path/to/TEP_data doc1.pdf doc2.pdf overview.md
```

### 4. Run the agent

```bash
python agent.py /path/to/TEP_data
```

```
> Analyze fault 6 and tell me what's going on
> What does the stripper actually do?
> Show me a chart of fault 6's pressure trend
```

## Project structure

| File | Purpose |
|---|---|
| `tep_loader.py` | Merges the 44 raw `.dat` files into labeled train/test parquet files |
| `rename_tep_columns.py` | Renames generic `xmeas_*`/`xmv_*` columns to real plant variable names |
| `twin_shap_explainer.py` | Standalone script: trains the twin, runs SHAP root-cause analysis for a given fault |
| `visualizations.py` | Reusable plotting functions (pressure trend, SHAP contributions) |
| `ingest_rag_corpus.py` | Chunks and embeds documentation for the RAG layer |
| `agent.py` | The Gemini-powered agent, wiring all tools together |
| `tep_plant_overview.md` | Plain-language plant explainer, part of the RAG corpus |

## Known limitations

- The digital twin currently predicts a single target variable (reactor
  pressure). Extending to a full multivariate twin is a natural next step.
- Fault labels in the raw test files apply to the entire file, but faults
  are only introduced partway through each run (~sample 160 of 960).
  Analysis in this project accounts for this by filtering to post-onset
  samples; be aware of this if extending the pipeline.
- The dataset provides limited samples per fault class (480-960 rows),
  which constrains statistical confidence, particularly for faults with
  historically subtle signatures (faults 3, 9, 15).

## Data & documentation sources

- Downs, J.J., Vogel, E.F. (1993). *A Plant-Wide Industrial Process
  Control Problem.* Computers & Chemical Engineering, 17(3), 245-255.
- Rieth et al. (2017), Tennessee Eastman Process Simulation Dataset,
  Harvard Dataverse.
- Duvall, P.M. (1996). *On-Line Optimization of the Tennessee Eastman
  Challenge Problem.* Texas Tech University thesis.
- Yapur, S.F. (2022). *Advantages of OKID-ERA Identification in Control
  Systems. An Application to the Tennessee Eastman Plant.* arXiv:2210.08538.

Raw dataset files and source PDFs are not included in this repository.
See Setup above for where to obtain them.
