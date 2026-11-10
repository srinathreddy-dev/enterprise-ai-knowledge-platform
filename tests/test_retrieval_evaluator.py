
from app.evaluation.retrieval_evaluator import RetrievalEvaluator
from app.ingestion.chunker import TextChunk
from app.evaluation.retrieval_metrics import EvaluationQuery


def create_chunks() -> list[TextChunk]:
    return [
        TextChunk(
            text="Employees can enroll in health insurance during open enrollment.",
            page_number=2,
            chunk_index=0,
            document_id="employee_handbook",
        ),
        TextChunk(
            text="Employees receive fifteen paid vacation days each year.",
            page_number=1,
            chunk_index=0,
            document_id="employee_handbook",
        ),
        TextChunk(
            text="The company provides retirement savings plans to eligible employees.",
            page_number=3,
            chunk_index=0,
            document_id="employee_handbook",
        ),
    ]


def assert_valid_metrics(evaluation):
    assert 0.0 <= evaluation.recall_at_1 <= 1.0
    assert 0.0 <= evaluation.recall_at_3 <= 1.0
    assert 0.0 <= evaluation.recall_at_5 <= 1.0
    assert 0.0 <= evaluation.mean_reciprocal_rank <= 1.0


def test_evaluator_calculates_bm25_metrics():
    evaluator = RetrievalEvaluator(create_chunks())

    evaluation = evaluator.evaluate_bm25(k=3)

    assert_valid_metrics(evaluation)


def test_evaluator_calculates_semantic_metrics():
    evaluator = RetrievalEvaluator(create_chunks())

    evaluation = evaluator.evaluate_semantic(k=3)

    assert_valid_metrics(evaluation)


def test_evaluator_calculates_hybrid_metrics():
    evaluator = RetrievalEvaluator(create_chunks())

    evaluation = evaluator.evaluate_hybrid(k=3)

    assert_valid_metrics(evaluation)


def test_evaluator_calculates_rrf_metrics():
    evaluator = RetrievalEvaluator(create_chunks())

    evaluation = evaluator.evaluate_rrf(k=3)

    assert_valid_metrics(evaluation)


def test_evaluator_accepts_custom_evaluation_queries():
    queries = [
        EvaluationQuery(
            query="health insurance",
            relevant_chunk_ids={
                "employee_handbook-page-2-chunk-0"
            },
        )
    ]

    evaluator = RetrievalEvaluator(
        create_chunks(),
        evaluation_queries=queries,
    )

    evaluation = evaluator.evaluate_bm25(k=1)

    assert evaluation.recall_at_1 == 1.0
    assert evaluation.recall_at_3 == 1.0
    assert evaluation.recall_at_5 == 1.0
    assert evaluation.mean_reciprocal_rank == 1.0

