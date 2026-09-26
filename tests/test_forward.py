import torch

from biotrace import BioTraceAgent, BioTraceEvidenceSpace, EvidenceStatus
from biotrace.agent import EvidenceObservation, EvidenceState


def test_synthetic_forward_shapes():
    torch.manual_seed(0)
    support = torch.tensor([[1, 0], [1, 1], [0, 1], [1, 0]], dtype=torch.float32)
    model = BioTraceEvidenceSpace(16, 8, 4, 2, 3, support)
    agent = BioTraceAgent(model, top_pathways=2, genes_per_pathway=1)
    sample = {
        "10x": torch.randn(1, 5, 16),
        "20x": torch.randn(1, 7, 16),
        "40x": torch.randn(1, 9, 16),
    }
    out = agent(sample)
    assert out["status"] in {"sufficient", "partial", "insufficient"}
    assert isinstance(out["evidence_score"], float)
    assert out["scale_cache"]["10x"]["gene_scores"].shape == (1, 4)
    assert out["scale_cache"]["20x"]["pathway_scores"].shape == (1, 2)
    assert out["scale_cache"]["40x"]["phenotype_scores"].shape == (1, 3)


def test_demo_verifier_can_stop_early_on_strong_current_evidence():
    state = EvidenceState()
    state.add(
        EvidenceObservation(
            source="synthetic_direct",
            magnification="10x",
            concept_index=0,
            patch_index=None,
            relevance=0.95,
            reliability_context=0.90,
            role="direct_phenotype",
        )
    )
    assert BioTraceAgent._demo_verify(state) == EvidenceStatus.SUFFICIENT
