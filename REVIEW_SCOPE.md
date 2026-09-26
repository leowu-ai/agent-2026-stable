# Review artifact scope

This repository is intentionally limited to a synthetic forward-pass artifact.
It is designed to expose the paper-level computation graph and evidence-state
interfaces while avoiding disclosure of patient/benchmark data, private storage
layout, trained weights, production prompts, or unreleased preprocessing code.

The following are intentionally **not** distributed in this snapshot:

- dataset download or preprocessing pipelines;
- benchmark questions, answers, labels, case identifiers, or train/test split files;
- raw WSIs or precomputed patient features;
- RNA-seq values, ssGSEA matrices, phenotype tables, or other patient-level supervision;
- model checkpoints and validation statistics;
- pathology knowledge-base contents and production RAG prompts;
- local model-serving scripts, IP addresses, filesystem paths, API credentials, or logs;
- benchmark prediction files and evaluation outputs;
- exact deployment heuristics beyond the algorithmic interfaces described in the paper.

`demo_forward.py` uses only random tensors and synthetic relation support. It is a
sanity-check for the forward data flow, not a substitute for the full experimental
pipeline used to report benchmark results.
