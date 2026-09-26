import torch

from demo_forward import build_pipeline


def test_full_synthetic_agent_forward():
    torch.manual_seed(0)
    pipeline = build_pipeline()
    pipeline.eval()
    features = {
        "10x": torch.randn(1, 12, 512),
        "20x": torch.randn(1, 24, 512),
        "40x": torch.randn(1, 48, 512),
    }
    with torch.no_grad():
        output = pipeline.forward(
            case_id="synthetic",
            question="Which synthetic phenotype is best supported?",
            choices=["Phenotype-A", "Phenotype-B", "Phenotype-C"],
            features_by_scale=features,
            question_feature=torch.randn(512),
        )

    assert output["answer"]["backend"] == "disabled"
    assert output["answer"]["answer_id"] is None
    memory = output["working_memory"]
    assert memory["observations"]
    assert memory["action_history"]
    assert memory["final_verifier"]["evidence_state"] in {
        "sufficient", "partial", "conflicting", "insufficient", "unavailable"
    }


def test_gene_pathway_membership_is_fixed_buffer():
    pipeline = build_pipeline()
    model = pipeline.g2p_agent.models["10x"]
    parameter_names = {name for name, _ in model.named_parameters()}
    buffer_names = {name for name, _ in model.named_buffers()}
    assert "H_prior" in buffer_names
    assert all("gene_pathway_strength" not in name for name in parameter_names)
