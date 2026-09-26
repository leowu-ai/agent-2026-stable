# BioTrace

**BioTrace** is a verifiable pathology agent for tracing molecular-to-phenotype
evidence in whole-slide VQA. This anonymized review snapshot exposes the core
architecture and inference interfaces described in the paper using a fully
synthetic forward example.

## What is included

The artifact mirrors the paper at the level needed to inspect the forward path:

- a multiscale Gene -> Pathway -> Phenotype evidence space;
- concept-specific prototype grounding and spatial responses;
- prior-constrained Gene -> Pathway support and trainable relation strengths;
- trainable Pathway -> Phenotype relations;
- independent 10x / 20x / 40x structured evidence computation;
- a lightweight evidence state with the five verifier states used by BioTrace;
- adaptive coarse-to-fine 10x -> 20x -> 40x acquisition without fixed multiscale fusion;
- optional Pathway/Gene support at 40x;
- a single random-tensor forward example and a small shape test.

The production system uses CONCH-derived WSI patch representations, Patho-R1-7B
for morphology-grounded visual observation, and Qwen3.5-9B for language-side
reasoning and evidence verification. Those external models and their deployment
wrappers are not required for the synthetic forward test below.

The lightweight verifier in this review artifact is explicitly a demo-only
controller. It recomputes evidence sufficiency after each acquisition so the
synthetic trajectory can stop early or continue to another magnification
depending on the currently accumulated evidence. It is not presented as the
production Qwen/RAG verifier used in the reported experiments.

## Synthetic forward pass

```bash
python -m pip install -r requirements.txt
python demo_forward.py
```

Expected output begins with:

```text
BioTrace synthetic forward: OK
```

The demo creates one synthetic WSI as random precomputed patch embeddings at
10x, 20x, and 40x and runs the evidence-space and agent-state forward path. It
does **not** load a WSI, patient record, benchmark question, answer, label, split,
checkpoint, or molecular measurement.

A minimal test is also provided:

```bash
python -m pytest -q
```

## Evidence semantics

Phenotype-, Pathway-, and Gene-level outputs are WSI-derived predictions. In the
full system, Pathway/Gene evidence is supportive context and is not interpreted
as measured RNA, IHC, FISH, mutation, copy number, or protein evidence. The
agent maintains predictive, visual, and biological evidence as distinguishable
sources throughout reasoning.

## Review-release boundary

This snapshot is intended for architecture inspection and forward-path
verification during double-blind review. It does not claim to be a standalone
package for reproducing the benchmark tables without the external datasets,
trained weights, and model services used by the full experimental pipeline.

To keep the review artifact free of patient/benchmark content and private
infrastructure, it deliberately omits dataset preparation, benchmark files and
labels, train/test split artifacts, patient-level supervision, trained weights,
production prompts, local service configuration, logs, and experimental outputs.
See `REVIEW_SCOPE.md` for the complete boundary.
