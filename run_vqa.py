#!/usr/bin/env python3
import argparse

from multiscale_vqa_agent import MultiScaleVQAPipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the G2P multi-agent pathology VQA pipeline"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--vqa_json", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no_crop", action="store_true")
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument(
        "--agent_mode",
        choices=("legacy", "hierarchical_rag"),
        default="legacy",
    )
    parser.add_argument("--knowledge_base", default=None)
    parser.add_argument(
        "--morphology_retrieval_mode",
        choices=("broad", "question_similarity"),
        default=None,
    )
    parser.add_argument(
        "--partial_retrieval_mode",
        choices=(
            "selected_phenotype",
            "question_similarity",
            "hybrid_question_prototype",
        ),
        default=None,
    )
    parser.add_argument(
        "--direct_retrieval_mode",
        choices=(
            "selected_phenotype",
            "question_similarity",
            "hybrid_question_prototype",
        ),
        default=None,
    )
    args = parser.parse_args()

    pipeline = MultiScaleVQAPipeline(
        args.config,
        morphology_retrieval_mode=args.morphology_retrieval_mode,
        partial_retrieval_mode=args.partial_retrieval_mode,
        direct_retrieval_mode=args.direct_retrieval_mode,
        agent_mode=args.agent_mode,
        knowledge_base=args.knowledge_base,
    )
    output = pipeline.run(
        vqa_path=args.vqa_json,
        output_path=args.output,
        limit=args.limit,
        crop_patches=not args.no_crop,
        resume=not args.no_resume,
        multiple_choice_only=True,
    )
    print(f"Saved answers: {output}")


if __name__ == "__main__":
    main()
