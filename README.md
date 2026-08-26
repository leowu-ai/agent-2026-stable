# G2P Pathology Agent

A clean public snapshot of our multi-agent whole-slide pathology reasoning framework for multiple-choice VQA.

This repository is organized from the model code at **`leowu-ai/agent2026@5d56466`**. Experimental outputs, benchmark answer files, local server scripts, private data paths, and model checkpoints are intentionally excluded.

## Overview

The framework combines structured **Gene–Program–Phenotype (G2P)** evidence with question-guided visual evidence from H&E whole-slide images (WSIs):

```text
Question + choices
        |
        v
Prototype-aware Planner / Router
        |
        +------------------------------+
        |                              |
        v                              v
Multi-scale G2P ToolBank        Knowledge RAG
(4096 / 2048 / 1024)                  |
        |                              |
        |                     scale-specific guidance
        |                              |
        +-----------> Evidence retrieval <----------+
                         |
                   4096 -> 2048 -> 1024
                         |
                         v
                  Patho-R1 visual observer
                         |
                         v
              Evidence Verifier + Working Memory
                         |
                         v
                Qwen Fusion / Arbitration
                         |
                         v
        answer + confidence + explanation + limitations
```

## G2P ToolBank

Each WSI is represented by patch features. Gene, program, and phenotype prototypes act as queries over patch embeddings through cross-attention:

```text
A = softmax(Q K^T / sqrt(d))
Z = A V
```

where `Q` is a gene/program/phenotype query, and `K,V` are projected WSI patch features. The resulting attention maps are used both for WSI-level representation and for retrieving high-attention patches.

The biological hierarchy is represented as:

```text
Gene -- H --> Program / Pathway -- R --> Phenotype
```

- `H`: fixed gene-to-program membership.
- `R`: program-to-phenotype relation weights initialized from biological priors and refined by the model.
- Gene/program evidence is **WSI-derived supportive evidence**, not a measured RNA, IHC, FISH, mutation, or protein assay.

See `models/g2p_toolbank.py` for the core model and `multiscale_vqa_agent/relation.py` for patient-aware relation reasoning.

## Evidence acquisition

The Agent separates two evidence roles:

1. **Structured G2P evidence** — target phenotype predictions, multi-scale agreement, validation reliability, and Gene–Program–Phenotype relations.
2. **Visual H&E evidence** — coarse-to-fine patch retrieval followed by Patho-R1 morphology observation.

The visual branch supports 4096 -> 2048 -> 1024 spatial refinement, cross-scale parent/child linking, attention-based phenotype/program/gene retrieval, question-similarity retrieval, and patch deduplication.

`KnowledgeRAG` supplies evidence semantics, limitations, candidate programs/genes, and scale-specific visual guidance. It does **not** directly provide the MCQ answer.

`EvidenceVerifierAgent` controls whether the system should inspect another scale, inspect supportive Program/Gene evidence, stop searching, or finalize. Final answer selection is performed only by the Qwen fusion/arbitration stage.

## Repository structure

```text
.
├── configs/
│   ├── agent.example.json        # sanitized runtime example
│   └── gene_programs.json        # biological Gene–Program–Phenotype definitions
├── data/
│   └── feature_io.py             # minimal patch-feature loader
├── models/
│   └── g2p_toolbank.py           # G2P prototype-attention model
├── multiscale_vqa_agent/
│   ├── agent_memory.py
│   ├── answerability.py
│   ├── clients.py
│   ├── fusion.py
│   ├── fusion_evidence.py
│   ├── g2p_runtime.py
│   ├── knowledge_rag.py
│   ├── pathology.py
│   ├── pipeline.py
│   ├── question_features.py
│   ├── registry.py
│   ├── relation.py
│   ├── retrieval.py
│   ├── schemas.py
│   ├── verifier.py
│   └── prompts/
│       └── fusion_arbiter.txt
└── run_vqa.py
```

## Requirements

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

For `.svs` WSI cropping, install the system OpenSlide library as well as `openslide-python`. Very large JPEG WSI cropping uses the `jpegtran` command-line utility.

The Qwen and Patho-R1 clients expect OpenAI-compatible inference endpoints (for example, local vLLM servers).

## Required external assets

The repository intentionally does not distribute patient data, WSI files, pretrained ToolBank checkpoints, frozen question embeddings, or the pathology knowledge-base ZIP. To run the full pipeline, provide:

- H&E WSI files;
- pre-extracted patch features for 1024 / 2048 / 4096 spatial scales;
- one trained `G2P_ToolBank_Minimal` directory per scale containing the checkpoint, vocabulary, relations, and tool metrics;
- an aligned feature manifest for each scale;
- optional frozen question features for question-similarity retrieval;
- a compatible pathology knowledge-base ZIP for `hierarchical_rag` mode;
- Qwen and Patho-R1 OpenAI-compatible endpoints.

No benchmark labels or answer archive is used as retrieval evidence by this public runtime.

## Configuration

Copy the example config and replace placeholder paths/endpoints:

```bash
cp configs/agent.example.json configs/agent.local.json
```

Important fields:

- `scales`: paths to the three trained G2P ToolBanks;
- `scale_manifests`: aligned feature manifests;
- `wsi_root`: WSI directory;
- `retrieval.question_feature_path`: optional frozen question embeddings;
- `qwen` and `pathor1`: OpenAI-compatible model endpoints.

Do not commit a local config if it contains private paths, addresses, or credentials.

## Run

Legacy evidence acquisition:

```bash
python run_vqa.py \
  --config configs/agent.local.json \
  --vqa_json /path/to/questions.json \
  --output outputs/answers.jsonl
```

Hierarchical RAG evidence acquisition:

```bash
python run_vqa.py \
  --config configs/agent.local.json \
  --vqa_json /path/to/questions.json \
  --output outputs/answers.jsonl \
  --agent_mode hierarchical_rag \
  --knowledge_base /path/to/pathology_knowledge_base.zip
```

Each input item should provide a case ID, question, and candidate choices using the original `Id` / `Question` / `Choice` convention (lower-case aliases are also supported internally).

## Notes on evidence semantics

- A phenotype Tool prediction is a WSI-derived prediction, not clinical ground truth.
- Program/Gene evidence is supportive and cannot be reinterpreted as a measured molecular assay.
- Patho-R1 is restricted to directly visible H&E morphology.
- RAG provides retrieval guidance and evidence-validity knowledge, not answer labels.
- Final Fusion arbitrates conflicting fallible evidence; it is not a simple majority vote.

## Source snapshot

Public organization snapshot derived from:

```text
repository: leowu-ai/agent2026
commit:     5d56466c24ba47a34d3ace68e38de30de0406379
message:    restore best runtime and add BCNB VQA adapter
```

The public repository is intentionally smaller than the research workspace so that the model architecture and inference pipeline can be inspected without unrelated experiment artifacts.
