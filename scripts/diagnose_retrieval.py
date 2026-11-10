
from pathlib import Path

from app.evaluation.dataset import EVALUATION_QUERIES
from app.ingestion.pipeline import DocumentIngestionPipeline
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.cross_encoder_reranker import CrossEncoderReranker
from app.retrieval.rrf_retriever import RRFHybridRetriever
from app.retrieval.semantic_search import SemanticSearcher


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIRECTORY = PROJECT_ROOT / "data" / "sample"


def shorten(text: str, limit: int = 220) -> str:
    text = " ".join(text.split())
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def main() -> None:
    pipeline = DocumentIngestionPipeline()
    documents = pipeline.process_directory(SAMPLE_DIRECTORY)

    chunks = [
        chunk
        for document in documents
        for chunk in document.chunks
    ]

    chunk_lookup = {
        chunk.chunk_id: chunk
        for chunk in chunks
    }

    bm25 = BM25Retriever(chunks)
    semantic = SemanticSearcher(chunks)

    rrf = RRFHybridRetriever(
        chunks,
        bm25_retriever=bm25,
        semantic_searcher=semantic,
        rrf_k=60,
        bm25_weight=0.70,
        semantic_weight=0.30,
    )

    print("\nLoading cross-encoder...")
    reranker = CrossEncoderReranker()

    print("\nPER-QUERY RETRIEVAL DIAGNOSTICS")
    print("=" * 90)

    for index, query in enumerate(EVALUATION_QUERIES, start=1):
        relevant_ids = query.relevant_chunk_ids

        bm25_results = bm25.search(query.query, top_k=5)
        rrf_results = rrf.search(query.query, top_k=10)

        reranked_results = reranker.rerank(
            query=query.query,
            chunks=[result.chunk for result in rrf_results],
            top_k=5,
        )

        bm25_ids = [
            result.chunk.chunk_id for result in bm25_results
        ]
        rrf_ids = [
            result.chunk.chunk_id for result in rrf_results[:5]
        ]
        reranked_ids = [
            result.chunk.chunk_id for result in reranked_results
        ]

        print(f"\nQUERY {index}: {query.query}")
        print("-" * 90)

        print("EXPECTED RELEVANT CHUNKS:")

        for chunk_id in sorted(relevant_ids):
            chunk = chunk_lookup.get(chunk_id)

            if chunk:
                print(f"  {chunk_id}")
                print(f"  Text: {shorten(chunk.text)}")
            else:
                print(f"  MISSING: {chunk_id}")

        print("\nTOP RESULT COMPARISON:")

        comparisons = [
            ("BM25", bm25_ids),
            ("Weighted RRF", rrf_ids),
            ("RRF + Cross-Encoder", reranked_ids),
        ]

        for name, result_ids in comparisons:
            first_id = result_ids[0] if result_ids else None
            hit = first_id in relevant_ids if first_id else False

            print(
                f"  {name:<24} "
                f"Top 1: {first_id} "
                f"| Correct: {hit}"
            )

        print("\nCROSS-ENCODER RANKING:")

        for rank, result in enumerate(reranked_results, start=1):
            correct = result.chunk.chunk_id in relevant_ids

            print(
                f"  {rank}. {result.chunk.chunk_id} "
                f"| Score: {result.score:.4f} "
                f"| Relevant: {correct}"
            )
            print(f"     {shorten(result.chunk.text)}")

    print("\n" + "=" * 90)
    print("Diagnostic completed successfully.")


if __name__ == "__main__":
    main()
