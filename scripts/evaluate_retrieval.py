
from pathlib import Path

from app.evaluation.retrieval_evaluator import RetrievalEvaluator
from app.ingestion.pipeline import DocumentIngestionPipeline


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIRECTORY = PROJECT_ROOT / "data" / "sample"


def main() -> None:
    pipeline = DocumentIngestionPipeline()
    documents = pipeline.process_directory(SAMPLE_DIRECTORY)

    all_chunks = [
        chunk
        for document in documents
        for chunk in document.chunks
    ]

    evaluator = RetrievalEvaluator(all_chunks)

    strategies = [
        ("BM25", evaluator.evaluate_bm25),
        ("Semantic Search", evaluator.evaluate_semantic),
        ("Weighted Hybrid", evaluator.evaluate_hybrid),
        ("Weighted RRF 70/30", evaluator.evaluate_rrf),
    ]

    results = []

    print("\nRETRIEVAL BENCHMARK")
    print("=" * 95)

    for name, evaluate in strategies:
        result = evaluate(k=5)
        results.append((name, result))

    print("\nLoading cross-encoder reranker...")

    reranked_result = evaluator.evaluate_rrf_with_reranking(
        k=5,
        candidate_k=10,
        bm25_weight=0.70,
        semantic_weight=0.30,
        rrf_k=60,
    )

    results.append(("RRF + Cross-Encoder", reranked_result))

    print(
        f"{'Strategy':<25}"
        f"{'Recall@1':<12}"
        f"{'Recall@3':<12}"
        f"{'Recall@5':<12}"
        f"{'MRR':<12}"
    )

    print("-" * 95)

    for name, result in results:
        print(
            f"{name:<25}"
            f"{result.recall_at_1:<12.4f}"
            f"{result.recall_at_3:<12.4f}"
            f"{result.recall_at_5:<12.4f}"
            f"{result.mean_reciprocal_rank:<12.4f}"
        )

    print("=" * 95)
    print("Retrieval benchmark completed successfully.")


if __name__ == "__main__":
    main()

