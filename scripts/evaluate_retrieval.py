
from pathlib import Path

from app.evaluation.retrieval_evaluator import RetrievalEvaluator
from app.ingestion.pipeline import DocumentIngestionPipeline


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PDF = PROJECT_ROOT / "data" / "sample" / "employee_handbook.pdf"


def main() -> None:
    if not SAMPLE_PDF.exists():
        raise FileNotFoundError(f"Sample PDF not found: {SAMPLE_PDF}")

    print("\nEnterprise AI Knowledge Platform")
    print("=" * 75)

    pipeline = DocumentIngestionPipeline()
    ingested_document = pipeline.process(SAMPLE_PDF)

    print(f"Document: {ingested_document.document.file_name}")
    print(f"Total chunks: {len(ingested_document.chunks)}")
    print("Initializing retrieval evaluation...")

    evaluator = RetrievalEvaluator(ingested_document.chunks)

    strategies = [
        ("BM25", evaluator.evaluate_bm25),
        ("Semantic Search", evaluator.evaluate_semantic),
        ("Hybrid Search", evaluator.evaluate_hybrid),
    ]

    print("\nRetrieval Evaluation Results")
    print("=" * 75)
    print(
        f"{'Strategy':<22}"
        f"{'Recall@1':>12}"
        f"{'Recall@3':>12}"
        f"{'MRR':>12}"
    )
    print("-" * 75)

    for strategy_name, evaluate in strategies:
        recall_at_1 = evaluate(k=1)
        recall_at_3 = evaluate(k=3)

        print(
            f"{strategy_name:<22}"
            f"{recall_at_1.recall_at_k:>12.3f}"
            f"{recall_at_3.recall_at_k:>12.3f}"
            f"{recall_at_3.mean_reciprocal_rank:>12.3f}"
        )

    print("=" * 75)
    print(f"Queries evaluated: {len(evaluator.evaluation_queries)}")
    print("Evaluation completed successfully.")


if __name__ == "__main__":
    main()

