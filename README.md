# RAG-GNN on Hetionet

![Pipeline overview](cover.png)

COMP 559 course project: heterogeneous GNN subgraph retrieval + LLM-based reasoning.

Pipeline: `Compound ←→ Disease` link prediction. The GNN mines the top-K meta-paths from Hetionet, verbalises them, and feeds them to Grok (xAI API). The LLM outputs a prediction + natural-language explanation.

---

## Quick Start

```bash
pip install -r requirements.txt

# 1. Download data + print statistics
python scripts/inspect_data.py

# 2. Train GNN (slow on CPU, fast on GPU)
python scripts/train_gnn.py --epochs 100 --ckpt checkpoints/gnn.pt

# 3. Train KGE baseline (DistMult / ComplEx)
python scripts/train_kge.py --kge distmult --epochs 30 --ckpt checkpoints/distmult.pt
python scripts/train_kge.py --kge complex  --epochs 30 --ckpt checkpoints/complex.pt

# 4. Evaluate Hits@K (GNN)
python scripts/eval_linkpred.py --ckpt checkpoints/gnn.pt

# 5. Set API key (Windows cmd)
set XAI_API_KEY=xai-xxx

# 6. Main experiment: GNN-only vs LLM-only vs GNN-RAG+LLM (+ DistMult) on the same test pairs, with McNemar test
python experiments/main_results.py --gnn-ckpt checkpoints/gnn.pt --kge-ckpt checkpoints/distmult.pt --n-pos 75 --n-neg 75 --judge

# 7. Error analysis: read runs/main_results.jsonl and print bucketed results
python experiments/error_analysis.py --file runs/main_results.jsonl

# 8. Top-K ablation
python experiments/ablation_k.py --ckpt checkpoints/gnn.pt --ks 0 1 3 5 10

# 9. DDR1 case study
python experiments/case_study_ddr1.py --ckpt checkpoints/gnn.pt
```

---

## Directory Structure

```
rag-gnn-hetionet/
├── cache/                        # Hetionet JSON cache
├── checkpoints/                  # Trained GNN checkpoints
├── runs/                         # RAG pipeline outputs
├── data/
│   ├── load_hetionet.py          # Download + build graph (PyG HeteroData)
│   └── splits.py                 # CtD edge train/val/test split + negative sampling
├── models/
│   ├── hetero_gnn.py             # SAGEConv + to_hetero heterogeneous GNN
│   └── link_predictor.py         # Dot-product / MLP scorer
├── retrieval/
│   ├── metapath.py               # 9 candidate meta-path templates (4 core + 4 co-regulation + 1 pathway-context)
│   ├── subgraph_extractor.py     # Path enumeration + GNN embedding scoring
│   └── verbalizer.py             # Path → natural language
├── llm/
│   ├── prompts.py                # System prompt + prediction / faithfulness prompt
│   └── client.py                 # xAI Grok API wrapper (requests)
├── scripts/
│   ├── inspect_data.py
│   ├── train_gnn.py              # Main training
│   ├── eval_linkpred.py          # AUROC / AUPRC / Hits@K
│   └── run_rag_pipeline.py       # End-to-end GNN-RAG + LLM
├── experiments/
│   ├── ablation_k.py             # Top-K impact
│   └── case_study_ddr1.py        # DDR1 qualitative analysis
└── requirements.txt
```

---

## Key Design Choices

| Item | Choice | Rationale |
|---|---|---|
| GNN backbone | `SAGEConv + to_hetero` | Simple, official PyG paradigm, supports 11 node types / 24 relation types in Hetionet |
| Node features | Learnable `nn.Embedding` | Hetionet has no node attributes; embeddings are learned from scratch |
| Main task | `Compound-treats-Disease` link prediction | 755 positive samples, classic drug-repurposing benchmark |
| Evaluation | AUROC / AUPRC / Hits@1/3/10 | All comparable to Rephetio baselines |
| Subgraph retrieval | 9 candidate meta-path templates + GNN cosine similarity scoring | Avoids path explosion, retains biological semantics |
| LLM | Grok 4 Fast Reasoning (xAI API) | Sufficient biomedical reasoning capability, low cost |
| Prompt structure | System + JSON schema output | Parseable; three fields: `prediction / confidence / rationale` |
| Faithfulness evaluation | LLM-as-judge (second call) | Checks whether the rationale hallucinates entities not in the retrieved paths |

---

## Experiment Checklist (data for the report)

| Experiment | Script | Output |
|---|---|---|
| Main results table (4 methods × N=150, with 95% CI + McNemar p-values) | `experiments/main_results.py --n-pos 75 --n-neg 75 --judge` | `runs/main_results.jsonl` + stdout table |
| KGE baseline AUROC/AUPRC | `scripts/train_kge.py --kge distmult` / `--kge complex` | stdout |
| GNN Hits@K | `scripts/eval_linkpred.py` | stdout |
| Top-K ablation | `experiments/ablation_k.py --ks 0 1 3 5 10` | `runs/ablation_k.json` |
| Faithfulness evaluation | `main_results.py --judge` (included automatically) | `judge` field in jsonl |
| Error analysis bucketing | `experiments/error_analysis.py` | stdout (bucketed by GNN-correct/LLM-wrong, etc.) |
| DDR1 case study | `experiments/case_study_ddr1.py` | stdout, qualitative |

**Recommended Experiments section structure for the report:**

1. **Main results table** (paste directly from `main_results.py` stdout) — four rows: GNN-only / LLM-only / GNN-RAG+LLM / DistMult, accuracy ± 95% CI
2. **McNemar significance** — proves GNN-RAG+LLM significantly outperforms LLM-only (supports "retrieval helps")
3. **Top-K ablation curve** — K=0 is LLM-only; shows where gains saturate as K increases
4. **Faithfulness** — proves the LLM does not hallucinate
5. **Error analysis (qualitative)** — typical GNN-correct/LLM-wrong cases + paths (shows pipeline limitations)
6. **DDR1 case study** — echoes the original proposal motivation

---

## Official Experiment Results (N=150, 75 pos + 75 neg, using Grok-4-Fast-Reasoning)

### Link Prediction AUROC (train_gnn/train_kge stdout)

| Model | Test AUROC | Test AUPRC |
|---|---|---|
| GNN (SAGEConv + to_hetero, 100 epochs) | **0.908** | 0.578 |
| DistMult (30 epochs) | 0.867 | 0.430 |

### Main Results Table (experiments/main_results.py + recompute_kge.py)

| Method | Accuracy | 95% CI |
|---|---|---|
| **GNN-RAG + LLM (ours)** | **0.8733** | [0.81, 0.92] |
| KGE (DistMult, calibrated) | 0.7867 | [0.72, 0.85] |
| GNN-only | 0.7267 | [0.65, 0.79] |
| LLM-only (no retrieval) | 0.5533 | [0.47, 0.63] |

### McNemar Paired Significance

- GNN-RAG+LLM **vs** LLM-only: p = 4.7 × 10⁻⁸ \*\*\*
- GNN-RAG+LLM **vs** GNN-only: p = 0.003 \*\*
- GNN-RAG+LLM **vs** KGE-calibrated: p = 0.04 \*

### Top-K Ablation (experiments/ablation_k.py, N=40)

| K | Accuracy |
|---|---|
| 0 (LLM-only) | 0.525 |
| 1 | 0.750 |
| 3 | 0.800 |
| **5** | **0.875** |
| 10 | 0.825 |

**K=5 is the saturation point.** K=10 actually decreases — too many paths dilute LLM attention.

### LLM Faithfulness

Rationale faithfulness rate: **52.3%** (n=107) — roughly half of rationales introduce entities outside the retrieved paths. Can be discussed in Limitations.

### Error Analysis Buckets (N=150)

| Category | Count |
|---|---|
| Both correct | 95 |
| GNN wrong but RAG+LLM correct (RAG rescue) | 36 |
| GNN correct but RAG+LLM wrong | 14 |
| Both wrong | 5 |
| retrieval_helped | **61** |
| retrieval_hurt | 13 |

Net retrieval benefit: 61 : 13 ≈ 4.7×.

---

## Known Limitations / Report Limitations Section

- **Closed-loop RL not implemented**: The original proposal's LLM-as-judge + RL retrieval strategy was not implemented (engineering scope exceeded budget). The current "static top-K + LLM faithfulness check" serves as groundwork for future work.
- **Manually enumerated meta-paths**: 9 candidate templates (4 core + 4 co-regulation + 1 pathway-context proxy); no automatic meta-path learning.
- **Small LLM evaluation sample**: Each pipeline run covers ~60 pairs (API cost); high variance.
- **No literature node features**: The original Hays paper used BioBERT-encoded PubMed abstracts as node features; this project does not include that layer (future work).
