
from pathlib import Path

from app.evaluation.retrieval_evaluator import RetrievalEvaluator
from app.ingestion.pipeline import DocumentIngestionPipeline


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PDF = PROJECT_ROOT / "data" / "sample" / "employee_handbook.pdf"


def main() -> None:
    if not SAMPLE_PDF.exists():
        raise FileNotFoundError(f"Sample PDF not found: {SAMPLE_PDF}")

    print("\nEnterprise AI Knowledge Platform")
    print("=" * 55)
    print("Loading and processing employee handbook...")

    ingestion_pipeline = DocumentIngestionPipeline()
    ingested_document = ingestion_pipeline.process(SAMPLE_PDF)

    print(f"Document: {ingested_document.document.file_name}")
    print(f"Total chunks: {len(ingested_document.chunks)}")

    print("\nInitializing retrieval evaluation...")
    evaluator = RetrievalEvaluator(ingested_document.chunks)

    strategies = [
        ("BM25", evaluator.evaluate_bm25),
        ("Semantic Search", evaluator.evaluate_semantic),
        ("Hybrid Search", evaluator.evaluate_hybrid),
    ]

    print("\nRetrieval Evaluation Results")
    print("=" * 55)
    print(f"{'Strategy':<22} {'Recall@3':>12} {'MRR':>12}")
    print("-" * 55)

    for strategy_name, evaluate in strategies:
        metrics = evaluate(k=3)

        print(
            f"{strategy_name:<22} "
            f"{metrics.recall_at_k:>12.3f} "
            f"{metrics.mean_reciprocal_rank:>12.3f}"
        )

    print("=" * 55)
    print("\nEvaluation completed successfully.")
    print(
        f"Queries evaluated: {len(evaluator.evaluation_queries)}"
    )


if __name__ == "__main__":
    main()

