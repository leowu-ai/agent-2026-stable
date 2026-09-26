# BioTrace

**BioTrace** is a pathology agent for evidence-guided whole-slide pathology VQA.
It organizes multiscale WSI representations into a structured
**Gene -> Pathway -> Phenotype** evidence space and performs question-driven
evidence acquisition and verification before generating an answer.

## Overview

BioTrace is built around three components:

1. **Structured biological evidence space.**  
   Concept-specific prototypes ground Gene, Pathway, and Phenotype concepts in
   multiscale WSI representations, producing patient-level predictions and
   spatial responses over patches.

2. **Adaptive evidence acquisition.**  
   The agent begins with compact phenotype evidence and progressively acquires
   additional visual evidence across **10x -> 20x -> 40x** when needed.
   Pathway- and Gene-level evidence can be queried at 40x as supportive
   biological context.

3. **Evidence verification and reasoning.**  
   The accumulated evidence is maintained in an explicit evidence state.
   After each acquisition step, the verifier updates the evidence status as
   `sufficient`, `partial`, `conflicting`, `insufficient`, or
   `unavailable`, allowing unresolved evidence requirements to guide the next
   observation.

The full BioTrace system uses **CONCH** for pathology representations,
**Patho-R1-7B** for morphology-grounded visual observation, and
**Qwen3.5-9B** for language-side reasoning and evidence verification.

## Evidence hierarchy

```text
Whole-Slide Image
       |
       v
Multiscale WSI representations
  10x      20x      40x
       |
       v
Gene -> Pathway -> Phenotype
       |
       v
Question-driven evidence acquisition
       |
       v
Evidence verification
       |
       v
Final answer
```

Phenotype-, Pathway-, and Gene-level outputs are inferred from WSI
representations. Pathway and Gene evidence provides supportive biological
context, while directly observed morphology and phenotype-level evidence remain
the primary evidence sources for pathology reasoning.

## Quick start

Install the minimal dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the included forward example:

```bash
python demo_forward.py
```

The example constructs random multiscale patch embeddings and runs them through
the BioTrace evidence space and adaptive evidence-acquisition path.

A minimal test suite is also included:

```bash
python -m pytest -q
```

## Repository structure

```text
.
├── biotrace/
│   ├── __init__.py
│   ├── agent.py
│   └── evidence_space.py
├── tests/
│   └── test_forward.py
├── demo_forward.py
├── requirements.txt
└── README.md
```

- `biotrace/evidence_space.py`: multiscale Gene-Pathway-Phenotype evidence modeling.
- `biotrace/agent.py`: evidence state, adaptive acquisition, and verification.
- `demo_forward.py`: end-to-end forward example.
- `tests/test_forward.py`: basic forward and adaptive-control tests.
