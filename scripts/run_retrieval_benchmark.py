
from pathlib import Path

from app.evaluation.dataset import EVALUATION_QUERIES
from app.evaluation.offline_embedder import OfflineEmbedder
from app.evaluation.retrieval_evaluator import RetrievalEvaluator
from app.ingestion.pipeline import DocumentIngestionPipeline


DATA_DIRECTORY = Path("data/sample")

TOP_K = 5

CANDIDATE_K = 10


def main() -> None:
    print("=" * 100)
    print("ENTERPRISE AI KNOWLEDGE PLATFORM")
    print("OFFLINE RETRIEVAL BENCHMARK")
    print("=" * 100)

    print("\nLoading evaluation documents...")

    documents = DocumentIngestionPipeline().process_directory(
        DATA_DIRECTORY
    )

    chunks = [
        chunk
        for document in documents
        for chunk in document.chunks
    ]

    available_chunk_ids = {
        chunk.chunk_id for chunk in chunks
    }

    missing_ids = {
        chunk_id
        for query in EVALUATION_QUERIES
        for chunk_id in query.relevant_chunk_ids
        if chunk_id not in available_chunk_ids
    }

    if missing_ids:
        raise ValueError(
            f"Evaluation labels reference missing chunks: {missing_ids}"
        )

    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks loaded: {len(chunks)}")
    print(f"Evaluation questions: {len(EVALUATION_QUERIES)}")
    print("Embedding mode: Local feature hashing")
    print("External API calls: None")

    print("\nInitializing retrieval evaluator...")

    evaluator = RetrievalEvaluator(
        chunks=chunks,
        evaluation_queries=EVALUATION_QUERIES,
        embedder=OfflineEmbedder(),
    )

    benchmark_results = [
        (
            "BM25",
            evaluator.evaluate_bm25(k=TOP_K),
        ),
        (
            "Offline feature-hash vectors",
            evaluator.evaluate_semantic(k=TOP_K),
        ),
        (
            "Hybrid",
            evaluator.evaluate_hybrid(k=TOP_K),
        ),
        (
            "Hybrid + Reranker",
            evaluator.evaluate_hybrid_with_reranking(
                k=TOP_K,
                candidate_k=CANDIDATE_K,
            ),
        ),
        (
            "Weighted RRF",
            evaluator.evaluate_weighted_rrf(
                k=TOP_K,
                bm25_weight=0.70,
                semantic_weight=0.30,
            ),
        ),
    ]

    print("\n" + "=" * 100)
    print("RETRIEVAL QUALITY RESULTS")
    print("=" * 100)

    print(
        f"{'Strategy':<30}"
        f"{'Recall@1':>10}"
        f"{'Recall@3':>10}"
        f"{'Recall@5':>10}"
        f"{'Precision@1':>13}"
        f"{'Precision@3':>13}"
        f"{'Precision@5':>13}"
        f"{'MRR':>10}"
    )

    print("-" * 112)

    for name, metrics in benchmark_results:
        print(
            f"{name:<30}"
            f"{metrics.recall_at_1:>10.3f}"
            f"{metrics.recall_at_3:>10.3f}"
            f"{metrics.recall_at_5:>10.3f}"
            f"{metrics.precision_at_1:>13.3f}"
            f"{metrics.precision_at_3:>13.3f}"
            f"{metrics.precision_at_5:>13.3f}"
            f"{metrics.mean_reciprocal_rank:>10.3f}"
        )

    print("-" * 112)

    print(
        "\nIMPORTANT: These are offline diagnostic results using "
        "feature hashing."
    )

    print(
        "They do not measure the quality of OpenAI production embeddings."
    )

    print(
        "The reranker is a lexical heuristic, not a trained cross-encoder."
    )


if __name__ == "__main__":
    main()
