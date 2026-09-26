# BioTrace

**BioTrace** is a verifiable pathology agent for tracing molecular-to-phenotype
evidence in whole-slide VQA. This repository is a compact, double-blind review
artifact that exposes the core forward computation on a fully synthetic sample.

## What is included

The artifact mirrors the paper at the level needed to inspect the forward path:

- a multiscale Gene -> Pathway -> Phenotype evidence space;
- concept-specific prototype grounding and spatial responses;
- prior-constrained Gene -> Pathway support and trainable relation strengths;
- trainable Pathway -> Phenotype relations;
- independent 10x / 20x / 40x structured evidence computation;
- a lightweight evidence state with the five verifier states used by BioTrace;
- coarse-to-fine 10x -> 20x -> 40x acquisition without fixed multiscale fusion;
- optional Pathway/Gene support at 40x;
- a single random-tensor forward example and a small shape test.

The production system uses CONCH-derived WSI patch representations, Patho-R1-7B
for morphology-grounded visual observation, and Qwen3.5-9B for language-side
reasoning and evidence verification. Those external models and their deployment
wrappers are not required for the synthetic forward test below.

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

To reduce data-leakage risk and keep the double-blind artifact free of private
infrastructure, this snapshot deliberately omits dataset preparation, benchmark
files and labels, train/test split artifacts, patient-level supervision, trained
weights, production prompts, local service configuration, logs, and experimental
outputs. See `REVIEW_SCOPE.md` for the complete boundary.

This repository is therefore a **forward-pass inspection artifact**, not a
standalone reproduction bundle for the reported benchmark numbers. Additional
training/evaluation assets can be released separately after the review process.
