# COMP 559 Project Complete Guide (for the team)

> Rice University — Machine Learning with Graphs — Spring 2026
> Team: Jingwen Chen, Jingtao Wang, Shouyu Wang
> This document is the technical overview and operation manual for all team members. Read this once and you are ready to start.

---

## Table of Contents

0. [One-Minute Summary](#0-one-minute-summary)
0.5. [Current Progress & Next Steps](#05-current-progress--next-steps)
1. [What is This Project](#1-what-is-this-project)
2. [From Original Proposal to Current Plan](#2-from-original-proposal-to-current-plan)
3. [Dataset — Hetionet in Detail](#3-dataset--hetionet-in-detail)
4. [Core Concepts (Required Reading)](#4-core-concepts-required-reading)
5. [System Architecture](#5-system-architecture)
6. [Code Map (What Each File Does)](#6-code-map-what-each-file-does)
7. [Environment Setup](#7-environment-setup)
8. [Full Run-Through Workflow](#8-full-run-through-workflow)
9. [Experiment Design (for the Report)](#9-experiment-design-for-the-report)
10. [Report Writing Skeleton (8 pages)](#10-report-writing-skeleton-8-pages)
11. [Division of Work](#11-division-of-work)
12. [Common Issues and Pitfalls](#12-common-issues-and-pitfalls)

---

## 0. One-Minute Summary

What we are doing:
- On **Hetionet** (a heterogeneous biomedical knowledge graph with 47,031 entities), we do **drug repurposing**: predicting which compound can treat which disease.
- The method has four layers: **heterogeneous GNN** learns node embeddings → **meta-path retrieval** finds interpretable paths from Compound to Disease → **verbalisation** → **LLM (Grok via xAI API)** generates a verdict + explanation based on those paths.
- We compare four methods: GNN-only, DistMult (KGE), LLM-only (no paths given), GNN-RAG+LLM (ours).
- Metrics: AUROC / AUPRC / Hits@K / paired accuracy / LLM explanation faithfulness.

The closed-loop LLM-as-judge + RL part from the original proposal was cut; it is left as future work. The simplification makes the project deliverable within 6 weeks.

**This project is at the upper-middle level for a graduate final project (A-/A)** — not a top-venue novelty push, but solid engineering and rigorous evaluation.

---

## 0.5 Current Progress & Next Steps

### Completed (as of 2026-04-19)

- [x] Design + documentation (this file)
- [x] Environment setup + Hetionet data loading (`scripts/inspect_data.py` runs successfully)
- [x] GNN training stack + smoke test (6 epochs → AUROC **0.897**)
- [x] DistMult KGE baseline code + smoke test (3 epochs → AUROC **0.8615**)
- [x] Retrieval stack (verified Cholecalciferol → osteoporosis path mining is correct)
- [x] LLM integration layer (xAI Grok API wrapper via requests)
- [x] Main experiment script (`main_results.py`, four-method comparison + McNemar + bootstrap CI)
- [x] Error analysis script + DDR1 case study + top-K ablation

### Next Steps (in order)

| Step | Command | Time | API key? | Output |
|---|---|---|---|---|
| 1 | `python scripts/train_gnn.py --epochs 100 --ckpt checkpoints/gnn.pt` | ~10 min | No | `gnn.pt` + real AUROC |
| 2 | `python scripts/train_kge.py --kge distmult --epochs 30 --ckpt checkpoints/distmult.pt` | ~10 min | No | `distmult.pt` + baseline AUROC |
| 3 | Register xAI account + top up $20 + get API key | ~5 min + $20 | — | env var configured |
| 4 | `python experiments/main_results.py --gnn-ckpt checkpoints/gnn.pt --kge-ckpt checkpoints/distmult.pt --n-pos 75 --n-neg 75 --judge` | ~15 min + ~$5-8 | Yes | **Main results table** + JSONL |
| 5 | `python experiments/ablation_k.py --ckpt checkpoints/gnn.pt --ks 0 1 3 5 10 --n-test 20 --n-neg 20` | ~10 min + ~$2 | Yes | **Top-K curve** |
| 6 | `python experiments/error_analysis.py --file runs/main_results.jsonl` | Seconds | No | **Error bucket table** + qualitative cases |
| 7 | `python experiments/case_study_ddr1.py --ckpt checkpoints/gnn.pt` | ~30 sec + 1 LLM call | Yes | **DDR1 qualitative analysis** |
| 8 | Full team writes 8-page report following §10 skeleton | ~1-2 weeks | — | Final deliverable (May 5) |

### Notes for Each Step

**Steps 1-2 (Training)**:
- No API key needed, free to run in the background
- After training, check `TEST AUROC` on the last line of stdout. The full training run should be 3-5 points higher than the smoke test (GNN 0.93+, DistMult 0.85+)
- If the number drops instead of rising, check for overfitting (early stopping is already in the code: it saves the checkpoint with the highest val AUROC)

**Step 3 (API key)**:
- Go to https://console.x.ai to register an xAI account
- Top up at least $5 (enough for the main experiment); $20 to be safe
- Get the `xai-...` key and set it as environment variable `XAI_API_KEY`
- Windows cmd: `set XAI_API_KEY=xai-xxx` (valid only in current session)
- Permanent: Windows System Properties → Environment Variables

**Step 4 (Main experiment)**:
- **The most critical step** — this produces all numbers in Table 1 of the report
- Before running the full experiment, do a quick sanity check: `--n-pos 5 --n-neg 5 --judge`, costing under $1, to verify there are no errors
- Then run the full N=150
- If budget is tight: add `--model grok-4-fast`, which reduces cost by 10× but lowers quality

**Step 5 (Ablation)**:
- For each K, run N=40 (20 positive + 20 negative), totalling 5 × 40 = 200 LLM calls
- K=0 is LLM-only (no paths), K=1/3/5/10 is GNN-RAG

**Steps 6-7 (Analysis)**:
- Cheap and must be run
- Error bucket output + DDR1 case are direct material for report §5 Case Study / Error Analysis

**Step 8 (Writing the report)**:
- Follow the 8-page skeleton in §10
- See §11 for division of work
- Use Overleaf for collaboration, GitHub for code
- Fill numbers and figures into Overleaf right after each experiment; don't let them pile up

### Budget Summary

| Item | Estimate |
|---|---|
| Grok 4 Fast Reasoning main experiment | ~$6 |
| ablation_k | ~$2 |
| DDR1 case + small experiments | ~$1 |
| Re-run buffer | ~$3 |
| **Total** | **~$12** |

Split ~$4 per person, or run everything under one account.

### Minimum Deliverable if Time is Tight (Last Week)

In priority order; cutting to this level still allows submission:

1. Step 1 (GNN training) — needed for GNN-only row in §4 main results
2. Step 4 (main experiment N=30) — minimum credible four-method comparison
3. Step 7 (DDR1 case) — one clean qualitative example
4. Report §3 Method + §4 main results table + §6 DDR1

Cut: DistMult (§4.3), ablation K (§4.2), error analysis (§5.1), faithfulness (§4.4).
This guarantees at least B+. Full plan targets A-.

---

## 1. What is This Project

### 1.1 One-Line Description

> Combining **heterogeneous graph neural networks** with **large language models** for interpretable drug-repurposing reasoning on a biomedical knowledge graph.

### 1.2 The Problem — Drug Repurposing

**Drug repurposing**: take an already-approved drug and find new therapeutic indications for it.

Why it matters:
- Developing a new drug takes an average of 10 years and $2 billion; approved drugs have already passed safety validation and can be deployed decades faster
- In 2020, remdesivir (originally an anti-Ebola drug) was repurposed for COVID-19

Why graphs help:
- If drug A and disease B share a path like "A binds protein G → G participates in pathway P → P is dysregulated in disease B," there is a pharmacological rationale
- Graph neural networks learn representations of nodes and their neighbourhoods, helping us quantify "how close A is to B"

### 1.3 Our Technical Approach

Four layers:
1. **GNN encoder**: learns a 128-dimensional vector for each node (compound, gene, disease, …)
2. **Subgraph retrieval**: given a (compound, disease) pair, enumerate all matching instances of pre-defined meta-paths of length ≤ 3 and select the top-K by GNN embedding similarity
3. **Verbalisation**: translate paths into English — "Aspirin binds PTGS2, PTGS2 is associated with inflammation"
4. **LLM reasoning**: Grok reads the paths and outputs JSON `{prediction, confidence, rationale}`

---

## 2. From Original Proposal to Current Plan

### 2.1 What the Original Proposal Aimed to Do

**Title**: Learning to Retrieve: Closed-loop GNN Subgraph Selection Optimized by LLM Feedback

**Original plan**:
- Data: cancer pathway subset of the STRING protein–protein interaction database, 379 proteins / 3498 edges
- LLM acts as a **judge** (referee), scoring retrieved subgraphs
- Use **reinforcement learning (PPO / REINFORCE)** with LLM scores as reward signals to train a "Learning to Retrieve" policy network
- DDR1 kinase as case study

### 2.2 Why We Simplified

| Original plan | Problem | Simplified version |
|---|---|---|
| Closed-loop RL retrieval policy | PPO requires difficult hyperparameter tuning; sparse rewards; not achievable in 6 weeks | Open-loop + static top-K |
| LLM as judge | Multiple calls + RL training, API budget blows up | LLM as generator, one forward pass |
| STRING 379-protein subset | Too small, single relation type (PPI only) | Hetionet: 47K nodes, 11 node types, 24 relation types |
| Custom literature corpus | High engineering cost | Entity names + relations in the graph already carry semantics |

### 2.3 What We Lost and What Remains

**Lost** (documented in Limitations + Future Work):
- Adaptive retrieval (currently fixed meta-paths + cosine similarity)
- Closed-loop optimisation (RL feedback)
- Literature corpus integration (BioBERT can be added later)

**Retained** (core selling points):
- GNN + LLM joint biomedical reasoning
- Interpretability: every prediction comes with a natural-language "why" chain
- Faithfulness evaluation: checks whether the LLM is hallucinating
- Full comparison: four-method paired significance testing

---

## 3. Dataset — Hetionet in Detail

### 3.1 What is Hetionet

- Open-source biomedical heterogeneous knowledge graph published by Himmelstein et al. in *eLife* (2017)
- Integrates 29 public databases (Entrez, DrugBank, MeSH, GO, SIDER, Reactome, …)
- **47,031 nodes, 2,250,197 edges**
- Widely used as a benchmark since 2017 (the Rephetio project)

Official repository: https://github.com/hetio/hetionet

### 3.2 Node Types and Counts (confirmed by our experiments)

| Node type | Count | Examples |
|---|---|---|
| Gene | 20,945 | DDR1, KRAS, TP53, BRCA1 |
| Biological Process | 11,381 | cell cycle, apoptosis |
| Side Effect | 5,734 | headache, nausea |
| Molecular Function | 2,884 | kinase activity |
| Pathway | 1,822 | PI3K signaling |
| **Compound** | **1,552** | Aspirin, Caffeine, Cholecalciferol |
| Cellular Component | 1,391 | nucleus, mitochondrion |
| Symptom | 438 | fever, fatigue |
| Anatomy | 402 | liver, brain |
| Pharmacologic Class | 345 | NSAIDs |
| **Disease** | **137** | type 2 diabetes, Alzheimer's, osteoporosis |

**Focus on Compound and Disease** — the two endpoints of our main task `Compound-treats-Disease`.

### 3.3 Relation Types

Our main task **`Compound-treats-Disease` (CtD)** has only **755 edges** (positive samples).
Because of the small count, the training set has only ~605 edges; overfitting is a real concern.

Other useful relations (used in retrieval and auxiliary tasks):

| Relation | Count | Interpretation |
|---|---|---|
| Compound-binds-Gene | 11,571 | Drug binds a target protein |
| Compound-upregulates-Gene | 18,756 | Drug upregulates gene expression |
| Compound-downregulates-Gene | 21,102 | Drug downregulates gene expression |
| Gene-associates-Disease | 12,623 | Gene linked to disease (GWAS, etc.) |
| Gene-interacts-Gene | 294,328 | Protein–protein interaction |
| Compound-resembles-Compound | 12,972 | Structural similarity between compounds |
| Compound-palliates-Disease | 390 | Drug alleviates symptoms (mild treatment) |
| Disease-localizes-Anatomy | 3,602 | Disease location in anatomy |

### 3.4 Why CtD is the Main Task

1. **Clear semantics**: "treats" is unambiguous
2. **Existing benchmark**: the Rephetio project reports AUROC ~0.97, giving us a reference
3. **High label quality**: sourced from DrugCentral + MEDI, clinically confirmed
4. **Very few samples (755)**: a feature not a bug — low-resource + small graph is exactly the setting where GNN + RAG complementarity shines

### 3.5 What Cases We Can Study

- **DDR1 + KRAS cancer case** (echoes original proposal): DDR1 is Gene[17750], KRAS is Gene[2523]; Hetionet has their association edges to multiple cancers
- **Vitamin D3 → osteoporosis**: we already verified that the top-1 retrieved path is biologically correct
- **Drug repurposing case**: find (compound, disease) pairs with high GNN prediction scores but no existing CtD edge, and manually check whether the prediction is plausible

---

## 4. Core Concepts (Required Reading)

### 4.1 Graph Neural Networks (GNN)

**One sentence**: a GNN lets each node update its representation by "talking to its neighbours."

**Standard message-passing formula**:

```
h_v^(l+1) = UPDATE( h_v^(l),  AGGREGATE({ h_u^(l) : u ∈ Neighbours(v) }) )
```

- `h_v^(l)`: vector representation of node v at layer l
- `AGGREGATE`: how to aggregate neighbour information (mean / sum / max / attention)
- `UPDATE`: combine own + neighbour information (typically an MLP)

**Effect of two GNN layers**: each node "sees" all information within 2 hops.

We use **GraphSAGE**:
```
h_v^(l+1) = σ(W · CONCAT(h_v^(l), MEAN({h_u^(l) : u ∈ N(v)})))
```

### 4.2 Heterogeneous Graphs and Meta-Paths

**Homogeneous vs heterogeneous graphs**:
- Homogeneous: all nodes/edges of one type (e.g., social network: user–friends–user)
- Heterogeneous: multiple types (Compound / Gene / Disease + various relations) — Hetionet is heterogeneous

**How heterogeneous GNNs work** (PyG's `to_hetero`):
- Train a separate set of message-passing weights for each relation type
- Each node collects messages from N different neighbour types, then combines them via sum/attention

**Meta-path**:
> A fixed template that switches between node types, defining a "semantic walk."

Examples (we define 9 in `retrieval/metapath.py`):
- `CbGaD`: Compound-binds-Gene-associates-Disease (gene bound by drug is associated with the disease)
- `CrCtD`: Compound-resembles-Compound-treats-Disease (a structurally similar drug treats this disease)
- `CbGiGaD`: Compound-binds-Gene-interacts-Gene-associates-Disease (an interacting partner of the drug target is associated with the disease)

A meta-path = a biological hypothesis. We use meta-paths to enumerate subgraphs, ensuring every retrieved path has a medical interpretation.

### 4.3 Link Prediction Task

**Task**: given two nodes (u, v) in the graph, predict whether a relation of a specific type exists between them.

Our example: given (Aspirin, Headache), predict whether the edge Aspirin-treats-Headache exists.

**Training procedure**:
1. Split CtD edges into train/val/test (605/75/75)
2. The training graph **only keeps train edges**; val/test edges are hidden
3. Model forward pass computes embeddings for all nodes
4. For each training positive (c, d), compute score `score(c, d) = h_c · h_d` (dot product)
5. Sample an equal number of negatives (c, d') with a target low score
6. BCE loss (cross entropy between sigmoid(score) and label)

**Evaluation**:
- **AUROC**: ranking quality; 1.0 is perfect, 0.5 is random
- **AUPRC**: more informative when positives are sparse (our positive:negative ≈ 1:10, imbalanced)
- **Hits@K**: given a disease, probability that the true positive compound appears in the top-K predictions
- **McNemar test**: paired significance test comparing two models on the same sample

### 4.4 Node Embeddings

**Embedding** = mapping discrete objects (words, nodes, users) to continuous vectors so that similar objects are close in vector space.

Hetionet nodes have no built-in features (unlike images with pixels or text with BERT representations), so we use **learnable `nn.Embedding`**:
- Each node is initialised with a random 128-dimensional vector
- During GNN training, these vectors are used as inputs and updated through message passing
- After training, each node has a "learned" representation

### 4.5 Knowledge Graph Embeddings (KGE) — DistMult and ComplEx

KGE was the dominant graph representation method before GNNs; instead of message passing, it learns scores for (head, relation, tail) triples directly.

**DistMult** (our baseline):
```
score(h, r, t) = h · diag(r) · t  = Σ_i h_i · r_i · t_i
```
- Each node and relation is a vector
- Training objective: high scores for positive triples, low scores for negatives

**Why include DistMult as a baseline**:
- Demonstrates whether "GNN message passing is better than simple node-relation embeddings" (or "not much better")
- Classic reference point in graph representation learning
- Reviewers / graders will question the paper without a KGE baseline

### 4.6 Retrieval-Augmented Generation (RAG)

**Original RAG idea** (Lewis et al. 2020):
1. User asks a question
2. A dense retriever fetches top-K relevant documents from a document store
3. Documents are concatenated into the prompt; the LLM generates an answer grounded in the documents
4. Benefit: the LLM doesn't need to "memorise" all knowledge; reduces hallucination; knowledge base can be updated

**GraphRAG / our variant**:
- The document store is replaced with **graph structure**
- Retrieval is not semantic-similarity top-K but **subgraph/path mining from the graph**
- The LLM generates an answer (or classification) based on the subgraph

**Why RAG is especially useful for biomedicine**:
- Base LLMs have incomplete knowledge about obscure drugs/genes
- Graph retrieval provides a "traceable evidence chain" — every conclusion can be traced back to specific edges
- Reduces LLM hallucination

### 4.7 The LLM's Role in the System (three options)

| Role | What it does | Our choice |
|---|---|---|
| **Generator** | Reads the subgraph, generates answer + explanation | **Yes** (main pipeline) |
| **Judge** | Scores model outputs, feeds back to training | Only for faithfulness evaluation |
| **Retriever** | Selects which subgraphs to feed downstream | Not used (RL part from original proposal was cut) |

Prompt structure (see [llm/prompts.py](llm/prompts.py)):
- System: role + output schema + constraint "only use entities from provided paths, do not hallucinate"
- User: compound name + disease name + list of paths
- Output: strict JSON `{prediction, confidence, rationale}`

### 4.8 Evaluation Metrics Summary

| Metric | Used for | Interpretation |
|---|---|---|
| AUROC | GNN / KGE link prediction | Ranking quality; 0.9+ is good |
| AUPRC | Same | More honest under class imbalance |
| Hits@K | GNN link prediction | Probability of true positive in top-K given a disease |
| Accuracy | LLM methods | Fraction of positive pairs predicted "yes" and negatives predicted "no" |
| 95% CI (bootstrap) | All metrics | Estimates metric stability |
| **McNemar** | Pairwise method comparison | p < 0.05 means method A is significantly better than B |
| **Faithfulness rate** | LLM explanation quality | Whether the LLM's rationale uses only entities from the provided paths |

---

## 5. System Architecture

### 5.1 Pipeline Overview

```
┌─────────────────────────────────────────────────────────────┐
│     Stage 0: Data (Hetionet JSON → PyG HeteroData)          │
│     11 node types, 47K nodes / 24 relation types, 2.25M edges│
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: GNN Training (SAGEConv + to_hetero)               │
│  Input:  HeteroData                                         │
│  Output: 128-dim embedding per node + link prediction score  │
│  Training objective: high score for CtD positives, low for  │
│                      negatives                              │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 2: Subgraph Retrieval                                │
│  Input:  (compound_idx, disease_idx), GNN embeddings        │
│  Enumerate all matching instances for each meta-path        │
│  Score by GNN embedding cosine similarity, select top-K     │
│  Output: K tuples of (nodes, edges, score)                  │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 3: Verbalisation                                     │
│  Each edge is rendered via a relation-specific template:    │
│  "Aspirin binds PTGS2 → PTGS2 is associated with ..."       │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 4: Grok API (xAI)                                    │
│  Input:  compound name, disease name, verbalised paths      │
│  Output: JSON {prediction: yes/no, confidence, rationale}   │
│  Optional: second Grok call for faithfulness judge          │
└─────────────────────────────────────────────────────────────┘
                           ↓
                  Final prediction + natural-language explanation
```

### 5.2 Four Comparison Methods in One Run

| Method | Procedure | Expected outcome |
|---|---|---|
| GNN-only | sigmoid(h_c · h_d) only | Baseline |
| DistMult (KGE) | h · r · t only | Traditional method baseline |
| LLM-only | Give drug and disease names only; LLM uses its own knowledge | Shows "LLM alone is not enough" |
| **GNN-RAG+LLM** | GNN retrieval + LLM reasoning (ours) | Should be highest |

---

## 6. Code Map (What Each File Does)

### 6.1 Data Layer

| File | Role |
|---|---|
| [data/load_hetionet.py](data/load_hetionet.py) | Downloads Hetionet JSON and parses it into PyG HeteroData. Stores node name and identifier. |
| [data/splits.py](data/splits.py) | Randomly splits CtD edges into train/val/test (80/10/10), **removes val/test edges from the training graph** (prevents leakage), provides negative sampling. |

### 6.2 Model Layer

| File | Role |
|---|---|
| [models/hetero_gnn.py](models/hetero_gnn.py) | Heterogeneous GNN. One `nn.Embedding` per node type, then 2 SAGEConv layers wrapped via `to_hetero`. |
| [models/link_predictor.py](models/link_predictor.py) | Link prediction head. `DotLinkPredictor` (dot product) and `MLPLinkPredictor` (concat + MLP). |
| [models/kge.py](models/kge.py) | Flattens HeteroData into (head, rel, tail) triples for DistMult/ComplEx. |

### 6.3 Retrieval Layer

| File | Role |
|---|---|
| [retrieval/metapath.py](retrieval/metapath.py) | 9 predefined meta-paths (CpD / CbGaD / CuGuD / CdGdD / CbGiGaD, etc.) |
| [retrieval/subgraph_extractor.py](retrieval/subgraph_extractor.py) | `build_adjacency`: converts edge_index to dict for fast neighbour lookup. `extract_paths`: enumerates all matching paths for a given (c, d) pair, scores them by GNN cosine similarity, returns top-K. |
| [retrieval/verbalizer.py](retrieval/verbalizer.py) | Natural-language templates for each relation type ("X binds Y", "X is associated with Y"), assembled into full path text. |

### 6.4 LLM Layer

| File | Role |
|---|---|
| [llm/prompts.py](llm/prompts.py) | System prompt (role + output schema + constraint), user prompt template, judge prompt. |
| [llm/client.py](llm/client.py) | xAI Grok API wrapper (direct requests). Provides `predict()` and `judge_faithfulness()`. |

### 6.5 Scripts Layer

| File | Role |
|---|---|
| [scripts/inspect_data.py](scripts/inspect_data.py) | Downloads data + prints all node/edge statistics. Run this first to verify data alignment. |
| [scripts/train_gnn.py](scripts/train_gnn.py) | Trains the heterogeneous GNN for CtD link prediction. `--epochs 100` takes ~5-10 min on GPU. |
| [scripts/train_kge.py](scripts/train_kge.py) | Trains DistMult or ComplEx baseline. `--epochs 30` takes ~10 min. |
| [scripts/eval_linkpred.py](scripts/eval_linkpred.py) | GNN-only evaluation: AUROC / AUPRC / Hits@1/3/10. |
| [scripts/run_rag_pipeline.py](scripts/run_rag_pipeline.py) | Early end-to-end version: GNN + retrieval + Grok. Main experiments use main_results.py instead. |

### 6.6 Experiments Layer

| File | Role |
|---|---|
| [experiments/main_results.py](experiments/main_results.py) | **Main experiment**. Runs GNN / DistMult / LLM-only / GNN-RAG+LLM on the same test pairs; outputs bootstrap CI + McNemar p-values. |
| [experiments/ablation_k.py](experiments/ablation_k.py) | Top-K ablation: K ∈ {0, 1, 3, 5, 10}; K=0 is LLM-only. |
| [experiments/case_study_ddr1.py](experiments/case_study_ddr1.py) | Qualitative analysis. Given a gene (default DDR1), finds the most embedding-similar (compound, disease) pair, extracts paths, has the LLM interpret them. |
| [experiments/error_analysis.py](experiments/error_analysis.py) | Reads main_results output and prints bucketed counts (GNN-correct/LLM-wrong, etc.) + representative cases. |

---

## 7. Environment Setup

### 7.1 Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU | Can run on CPU, but ~10× slower | ≥ 8 GB VRAM (RTX 3060 or better) |
| RAM | 8 GB | 16 GB |
| Disk | 1 GB (data + dependencies) | 2 GB |

The full Hetionet graph fits in < 2 GB GPU memory — very manageable.

### 7.2 Software Installation (Windows + Anaconda)

```bash
cd "c:/Users/10249/OneDrive/Desktop/RAG/rag-gnn-hetionet"
pip install -r requirements.txt
```

`requirements.txt` contents:
```
torch_geometric>=2.5.0
networkx>=3.0
requests>=2.31
tqdm>=4.66
(only requests is needed beyond what Anaconda base already has)
```

**Note**: do not reinstall `torch`. Anaconda base already includes PyTorch 2.8 dev + CUDA 12.8, which is sufficient.

Mac/Linux also uses `pip install -r requirements.txt` — no difference.

### 7.3 API Key Configuration

Register an xAI account and get an API key: https://console.x.ai

Windows cmd:
```cmd
set XAI_API_KEY=xai-xxxxxxxxxxxx
```

Windows PowerShell:
```powershell
$env:XAI_API_KEY="xai-xxxxxxxxxxxx"
```

macOS/Linux bash:
```bash
export XAI_API_KEY=xai-xxxxxxxxxxxx
```

Permanent storage: on Windows, add via System Properties → Environment Variables. On macOS, add to `~/.zshrc`.

You can also create a `.env` file (see `.env.example`), but the code currently reads only from environment variables.

### 7.4 Data Download

The first time `scripts/inspect_data.py` is run, it automatically downloads the Hetionet JSON (16.1 MB) from GitHub and caches it at `cache/hetionet-v1.0.json.bz2`. Subsequent runs use the cache.

If the network is unavailable:
- Use a VPN or run on a machine outside the firewall
- Or manually curl/wget the file into `cache/`

---

## 8. Full Run-Through Workflow

### 8.1 Training Phase (one-time, ~20 min)

```bash
# Verify environment
python scripts/inspect_data.py

# Train GNN
python scripts/train_gnn.py --epochs 100 --ckpt checkpoints/gnn.pt

# Train KGE baseline (DistMult required, ComplEx optional)
python scripts/train_kge.py --kge distmult --epochs 30 --ckpt checkpoints/distmult.pt
python scripts/train_kge.py --kge complex  --epochs 30 --ckpt checkpoints/complex.pt
```

After training, `checkpoints/` should contain `gnn.pt` and `distmult.pt` (optionally `complex.pt`).

### 8.2 Evaluation Phase

```bash
# GNN metrics
python scripts/eval_linkpred.py --ckpt checkpoints/gnn.pt

# Expected output:
# AUROC  0.93xx
# Hits@1  0.xx
# Hits@3  0.xx
# Hits@10 0.xx
```

### 8.3 Experiment Phase (requires API key)

```bash
set XAI_API_KEY=xai-xxx

# Main experiment: four-method comparison, N=150 (75 pos + 75 neg), with faithfulness judge
python experiments/main_results.py ^
    --gnn-ckpt checkpoints/gnn.pt ^
    --kge-ckpt checkpoints/distmult.pt ^
    --n-pos 75 --n-neg 75 --judge

# Output goes to runs/main_results.jsonl; stdout shows four-method accuracy table + McNemar p-values

# Top-K ablation (medium scale N=40)
python experiments/ablation_k.py --ckpt checkpoints/gnn.pt --ks 0 1 3 5 10 --n-test 20 --n-neg 20

# DDR1 case study
python experiments/case_study_ddr1.py --ckpt checkpoints/gnn.pt

# Error analysis (reads main_results.jsonl)
python experiments/error_analysis.py --file runs/main_results.jsonl --per-bucket 5
```

### 8.4 Full Shell Script (one-click run)

Save as `run_all.sh` (run in Git Bash on Windows):

```bash
#!/bin/bash
set -e

echo "== Step 1: inspect data =="
python scripts/inspect_data.py | tail -5

echo "== Step 2: train GNN (100 epochs) =="
python scripts/train_gnn.py --epochs 100 --ckpt checkpoints/gnn.pt

echo "== Step 3: train DistMult (30 epochs) =="
python scripts/train_kge.py --kge distmult --epochs 30 --ckpt checkpoints/distmult.pt

echo "== Step 4: main experiment (needs API key) =="
python experiments/main_results.py \
    --gnn-ckpt checkpoints/gnn.pt \
    --kge-ckpt checkpoints/distmult.pt \
    --n-pos 75 --n-neg 75 --judge

echo "== Step 5: ablation K =="
python experiments/ablation_k.py --ckpt checkpoints/gnn.pt --ks 0 1 3 5 10 --n-test 20 --n-neg 20

echo "== Step 6: error analysis =="
python experiments/error_analysis.py --file runs/main_results.jsonl --per-bucket 5

echo "== Step 7: DDR1 case =="
python experiments/case_study_ddr1.py --ckpt checkpoints/gnn.pt
```

---

## 9. Experiment Design (for the Report)

### 9.1 Experiment Checklist (by report section)

| Report section | Experiment | Script | Expected output |
|---|---|---|---|
| §4.1 Main results table | N=150 four methods | `main_results.py` | Table 1 with accuracy ± CI + significance stars |
| §4.2 Top-K ablation | K ∈ {0,1,3,5,10} | `ablation_k.py` | Line chart + saturation point analysis |
| §4.3 KGE comparison | DistMult / ComplEx | `train_kge.py` | Two rows added to main table |
| §4.4 Faithfulness | `--judge` flag | `main_results.py` | Single number + a few invented-entity examples |
| §5.1 Error analysis | Four buckets | `error_analysis.py` | Table + qualitative case descriptions |
| §5.2 DDR1 case | Qualitative analysis | `case_study_ddr1.py` | 3 paths + LLM rationale |

### 9.2 Expected Results

| Method | Expected accuracy | Notes |
|---|---|---|
| GNN-only | 0.80–0.85 | Upper bound for small-sample link prediction |
| DistMult | 0.75–0.82 | Pure KGE is usually slightly weaker than GNN |
| LLM-only (no retrieval) | 0.55–0.65 | Grok has limited knowledge of obscure compounds |
| **GNN-RAG+LLM** | **0.85–0.92** | Should be highest |

McNemar: GNN-RAG+LLM vs LLM-only should give p < 0.001 (retrieval significantly helps).
Faithfulness: should be > 80% (prompt constraint is effective).

### 9.3 Budget Estimate

| Item | Count | Unit cost estimate | Subtotal |
|---|---|---|---|
| Main experiment (N=150 × 2 LLM calls + optional judge 150 calls) | ≈450 | Grok 4 Fast Reasoning ~$0.01/call (with caching) | ~$5 |
| Top-K ablation (5 K values × 40 samples) | 200 | Same | ~$2 |
| One re-run for backup data | — | — | ~$3 |
| **Total** | | | **$10** |

If budget is tight:
- Switch `main_results.py` to `--model grok-4-fast`, reducing cost by 10×
- Reduce N from 150 to 100

---

## 10. Report Writing Skeleton (8 pages)

### 10.1 Abstract (half page)
- 1 sentence motivation: GNN excels at topology but lacks semantic explanation
- 2 sentences method: GNN retrieval + LLM reasoning + faithfulness verification
- 2 sentences results: four-method comparison + McNemar significance + case study
- 1 sentence contribution

### 10.2 Introduction (1 page)
- Importance of drug repurposing
- Both graph structure and literature semantics matter
- Existing RAG-GNN is open-loop (cite [3] Hays & Richardson)
- Our contributions: three points (the ones still valid after cutting the closed loop)
  1. Complete GraphRAG pipeline on Hetionet
  2. Four-method comparison + paired significance evaluation
  3. Faithfulness evaluation (hot topic in RAG)

### 10.3 Related Work (0.5 page)
- GraphRAG / GNN-RAG (cite Mavromatis 2024)
- KG-aware LLM (ToG, KG-GPT)
- Drug repurposing on Hetionet (Himmelstein 2017, Rephetio)
- LLM-as-judge (Zheng 2023)

### 10.4 Method (2 pages)
- §3.1 Problem formulation
- §3.2 Hetero GNN encoder (SAGEConv + to_hetero, with formula)
- §3.3 Link predictor (dot product)
- §3.4 Meta-path subgraph retrieval (list 9 meta-paths, scoring formula)
- §3.5 Path verbalisation (templates)
- §3.6 LLM reasoning (prompt structure + JSON schema)
- §3.7 Faithfulness judge

### 10.5 Experiments (2.5 pages)
- §4.1 Dataset (Hetionet stats table)
- §4.2 Setup (hyperparameter table: hidden=128, layers=2, lr=1e-3, epochs=100)
- §4.3 Main results (main table + McNemar)
- §4.4 Ablation K
- §4.5 Faithfulness
- §4.6 Error analysis (bucket table + 2 qualitative cases)

### 10.6 Case Study: DDR1 (0.5 page)
- Connects to original proposal motivation: DDR1 kinase + KRAS + cancer
- Show 3 retrieved paths + LLM interpretation

### 10.7 Discussion & Limitations (0.5 page)
- Meta-paths are hand-designed (no automatic mining)
- No literature node features (BioBERT not used)
- LLM cost limits N=150

### 10.8 Future Work (0.25 page)
- Closed-loop LLM-as-judge with RL
- Adding BioBERT literature features
- Automatic meta-path mining
- Information-theoretic decomposition to quantify GNN/LLM complementary information (echoes proposal [1][3])

### 10.9 Conclusion (0.25 page)
- 3-sentence summary

### 10.10 References
- Hetionet (Himmelstein 2017)
- GNN-RAG (Mavromatis 2024)
- RAG-GNN (Hays & Richardson 2026) — proposal [3]
- DistMult (Yang 2015), ComplEx (Trouillon 2016)
- GraphSAGE (Hamilton 2017)
- Grok (xAI technical report)
- Bertschinger et al. (2014) — proposal [1]
- Zheng et al. (2023) — proposal [6]

---

## 11. Division of Work (3-person team)

Assuming ~3 weeks remain until the May 5 deadline:

### Jingwen Chen (first author, main writer)
- Run all training and main experiments (train_gnn / train_kge / main_results)
- Write Abstract, Introduction, Method sections
- Final integration of all three members' writing

### Jingtao Wang (experimental analysis)
- Run ablation_k and error_analysis; prepare tables and figures
- Write Experiments section
- Design the specific narrative for the DDR1 case study

### Shouyu Wang (method details + future work)
- Study metapath and verbalizer; optionally add 1-2 more complex meta-paths
- Run case_study_ddr1 and manually curate the best 3 paths
- Write Related Work, Discussion, Future Work sections

### Collaboration Cadence
- Weekly sync to review results and report progress
- Private GitHub repo for code, Overleaf for the report
- Fill numbers and figures into Overleaf immediately after each experiment — do not let them pile up

---

## 12. Common Issues and Pitfalls

### 12.1 Installation Issues

**Q: `pip install torch_geometric` fails**
A: If the Anaconda base torch is an older version it may be incompatible. Check with `python -c "import torch; print(torch.__version__)"`. Requires >= 2.0.

**Q: CUDA not available**
A: Run `python -c "import torch; print(torch.cuda.is_available())"`. If False, you may need to reinstall PyTorch with CUDA support. Or just run on CPU — 5-10× slower but functional.

**Q: `ModuleNotFoundError: data.load_hetionet`**
A: You must run `python scripts/xxx.py` from the project root (`rag-gnn-hetionet/`), not from inside `scripts/`. Each script already has `sys.path.insert` to handle this.

### 12.2 Training Issues

**Q: `UserWarning: The type 'Molecular Function' contains invalid characters`**
A: Does not affect training. PyG complains about spaces in node type names. To suppress it, rename "Molecular Function" to "MolecularFunction" in `load_hetionet.py` and update all references.

**Q: Training loss does not decrease / AUROC does not improve**
A: Check that `split.train_pos.size(1)` equals 605. If it is 0, there is a data loading problem.

**Q: Out of GPU memory**
A: Use `--hidden 64`, or run on CPU with `--device cpu`.

### 12.3 API Issues

**Q: xAI API 401 / "Incorrect API key"**
A: The key was not set correctly. Run `echo %XAI_API_KEY%` (Windows cmd) to confirm the value is visible. Or the key may have been revoked — create a new one at console.x.ai.

**Q: `rate_limit_error`**
A: xAI rate limits are fairly generous. The scripts make serial calls so this should rarely trigger. If it does, add `time.sleep(1)` between calls in llm.predict.

**Q: JSON parse failure**
A: `_parse_json` in llm/client.py uses a regex to extract the first `{...}` block. If the LLM outputs extra text, the exception is caught and "no" + confidence 0 is returned. Occasional failures do not affect overall results.

**Q: API costs spiralling**
A: Switch to `--model grok-4-fast`, which is 10× cheaper. Or reduce N.

### 12.4 Report Issues

**Q: Main experiment N=150, but positive and negative samples are unbalanced**
A: Our default is 75 positive + 75 negative (1:1 balanced), so accuracy is a valid metric. For a more realistic imbalanced scenario, use `--n-neg 750` (1:10 ratio) and report AUPRC instead.

**Q: LLM-only accuracy is unexpectedly high (0.80 instead of 0.55)**
A: Grok has strong knowledge of well-known drugs and diseases. This actually makes a good story: retrieval benefit is largest for obscure compounds. Stratify by compound frequency in the report — "retrieval benefit is largest for rare compounds."

**Q: McNemar p-value is large (not significant)**
A: Either N is too small (increase to N=200) or the methods genuinely perform similarly (report it as a finding in Discussion, not a bug).

**Q: Faithfulness rate is very low (< 70%)**
A: Strengthen the "only use entities from provided paths" constraint in the system prompt. Consider adding a self-correction step: ask the LLM to first list the entities it used, then verify they all appear in the paths.

---

## Appendix A: Quick Command Reference

```bash
# Environment
pip install -r requirements.txt

# Data
python scripts/inspect_data.py

# Training
python scripts/train_gnn.py --epochs 100 --ckpt checkpoints/gnn.pt
python scripts/train_kge.py --kge distmult --epochs 30 --ckpt checkpoints/distmult.pt

# Evaluation
python scripts/eval_linkpred.py --ckpt checkpoints/gnn.pt

# Main experiment
python experiments/main_results.py --gnn-ckpt checkpoints/gnn.pt --kge-ckpt checkpoints/distmult.pt --n-pos 75 --n-neg 75 --judge

# Ablation
python experiments/ablation_k.py --ckpt checkpoints/gnn.pt --ks 0 1 3 5 10

# Analysis
python experiments/error_analysis.py --file runs/main_results.jsonl
python experiments/case_study_ddr1.py --ckpt checkpoints/gnn.pt
```

## Appendix B: Key Parameter Reference

| Parameter | Default | Where to set |
|---|---|---|
| GNN hidden dim | 128 | `train_gnn.py --hidden` |
| GNN layers | 2 | `train_gnn.py --layers` |
| GNN epochs | 100 | `train_gnn.py --epochs` |
| Learning rate | 1e-3 | `train_gnn.py --lr` |
| Negative sample ratio | 1:1 | `train_gnn.py --neg-ratio` |
| Retrieval top-K | 5 | `main_results.py --top-k` |
| Main experiment N | 75+75 | `main_results.py --n-pos --n-neg` |
| LLM model | grok-4-fast-reasoning | `--model` |

---

**Check §12 for any issues before asking the group.**
