from app.evaluation.retrieval_evaluator import RetrievalEvaluator
from app.ingestion.chunker import TextChunk


def create_chunks() -> list[TextChunk]:
    """Create the chunks used by the retrieval evaluation."""
    return [
        TextChunk(
            text="Employees can enroll in health insurance during open enrollment.",
            page_number=1,
            chunk_index=0,
        ),
        TextChunk(
            text="Employees receive fifteen paid vacation days each year.",
            page_number=2,
            chunk_index=1,
        ),
        TextChunk(
            text="The company provides retirement savings plans to eligible employees.",
            page_number=3,
            chunk_index=2,
        ),
    ]


def test_evaluator_calculates_bm25_metrics():
    evaluator = RetrievalEvaluator(create_chunks())

    evaluation = evaluator.evaluate_bm25(k=3)

    assert evaluation.recall_at_k == 1.0
    assert evaluation.mean_reciprocal_rank == 1.0


def test_evaluator_calculates_semantic_metrics():
    evaluator = RetrievalEvaluator(create_chunks())

    evaluation = evaluator.evaluate_semantic(k=3)

    assert 0.0 <= evaluation.recall_at_k <= 1.0
    assert 0.0 <= evaluation.mean_reciprocal_rank <= 1.0


def test_evaluator_calculates_hybrid_metrics():
    evaluator = RetrievalEvaluator(create_chunks())

    evaluation = evaluator.evaluate_hybrid(k=3)

    assert 0.0 <= evaluation.recall_at_k <= 1.0
    assert 0.0 <= evaluation.mean_reciprocal_rank <= 1.0


def test_evaluator_accepts_custom_evaluation_queries():
    from app.evaluation.retrieval_metrics import EvaluationQuery

    queries = [
        EvaluationQuery(
            query="health insurance",
            relevant_chunk_indexes={0},
        )
    ]

    evaluator = RetrievalEvaluator(
        create_chunks(),
        evaluation_queries=queries,
    )

    evaluation = evaluator.evaluate_bm25(k=1)

    assert evaluation.recall_at_k == 1.0
    assert evaluation.mean_reciprocal_rank == 1.0
