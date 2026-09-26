# BioTrace

**BioTrace** is a pathology agent for evidence-guided whole-slide pathology VQA.
It organizes multiscale WSI representations into a structured
**Gene -> Pathway -> Phenotype** evidence space and performs question-driven
evidence acquisition and verification before final answer generation.

## Overview

BioTrace follows three coupled stages:

1. **Evidence requirement.** The Planner converts a pathology question into a
   structured evidence requirement, including the target phenotype, requested
   evidence type, preferred observation scale, and answer constraint.
2. **Evidence acquisition.** Scale-specific structured models expose phenotype,
   Pathway, and Gene predictions together with concept-specific spatial responses.
   The Agent adaptively acquires direct phenotype evidence, H&E morphology, and
   supportive biological evidence across 10x, 20x, and 40x.
3. **Evidence verification.** After each acquisition, the Evidence Verifier
   updates the evidence state and determines whether the current evidence is
   sufficient or another observation is required.

The full system uses **CONCH** pathology representations, **Patho-R1-7B** as the
morphology observer, and **Qwen3.5-9B** for language-side planning, verification,
and evidence arbitration.

## Structured evidence space

At each magnification, concept-specific prototypes ground Gene, Pathway, and
Phenotype concepts in WSI patch features. Gene evidence is aggregated into
Pathway context using biologically defined Gene -> Pathway membership, while
Pathway -> Phenotype relation strengths connect biological programs to phenotype
representations.

The forward path keeps the same evidence semantics used by the Agent:

```text
Gene evidence
     |
     v
Pathway evidence
     |
     v
Phenotype evidence
     |
     +--------------------+
     |                    |
     v                    v
spatial responses     patient-level predictions
     |                    |
     +----------+---------+
                |
                v
       evidence acquisition
```

Pathway- and Gene-level values are WSI-derived predictions and are treated as
supportive evidence. They are not interpreted as measured RNA, IHC, FISH/ISH,
mutation, amplification, copy number, or protein assays.

## Agent loop

```text
Question + choices
       |
       v
Evidence Planner
       |
       v
Round-0 phenotype evidence (10x)
       |
       v
Evidence Verifier
       |
       +---- sufficient ----------------------> Final Arbiter
       |
       +---- unresolved
       v
10x -> 20x -> 40x morphology acquisition
       |
       v
optional Pathway / Gene support at 40x
       |
       v
Evidence Verifier -> Final Arbiter
```

The working memory keeps direct and supportive evidence separate and records the
acquisition trajectory. The verifier uses five evidence states:
`sufficient`, `partial`, `conflicting`, `insufficient`, and `unavailable`.

## Prompt interfaces

The main language-side contracts are included under
`multiscale_vqa_agent/prompts/`:

- `planner.txt`: question -> evidence requirement;
- `pathology_observer.txt`: morphology-only visual observation;
- `verifier.txt`: evidence sufficiency and next-action selection;
- `fusion_arbiter.txt`: final evidence arbitration.

These prompts encode the evidence roles and constraints used by the agent while
keeping the final answer grounded in the accumulated evidence state.

## Synthetic forward example

Install the minimal dependencies:

```bash
python -m pip install -r requirements.txt
```

Run one end-to-end forward pass:

```bash
python demo_forward.py
```

The example creates random precomputed patch embeddings at 10x, 20x, and 40x,
then runs the Planner, structured evidence models, relation reasoning, patch
retrieval, working memory, Evidence Verifier, and final arbitration interfaces.

A small test suite is included:

```bash
python -m pytest -q
```

## Repository structure

```text
.
├── configs/
│   └── agent.example.json
├── models/
│   ├── __init__.py
│   └── g2p_toolbank.py
├── multiscale_vqa_agent/
│   ├── __init__.py
│   ├── agent_memory.py
│   ├── clients.py
│   ├── fusion.py
│   ├── g2p_runtime.py
│   ├── knowledge_rag.py
│   ├── pathology.py
│   ├── question_features.py
│   ├── pipeline.py
│   ├── planner.py
│   ├── registry.py
│   ├── relation.py
│   ├── retrieval.py
│   ├── schemas.py
│   ├── verifier.py
│   └── prompts/
│       ├── planner.txt
│       ├── pathology_observer.txt
│       ├── verifier.txt
│       └── fusion_arbiter.txt
├── tests/
│   └── test_forward.py
├── demo_forward.py
├── requirements.txt
└── README.md
```
