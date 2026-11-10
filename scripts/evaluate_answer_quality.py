
from dataclasses import dataclass
from pathlib import Path

from app.evaluation.answer_quality_evaluator import (
    AnswerQualityEvaluator,
)
from app.evaluation.semantic_evaluator import (
    SemanticFaithfulnessEvaluator,
)
from app.ingestion.pipeline import DocumentIngestionPipeline
from app.reranking.cross_encoder_reranker import RerankedResult
from app.retrieval.bm25_retriever import (
    BM25Retriever,
    RetrievalResult,
)
from app.retrieval.semantic_retriever import SemanticEmbedder


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIRECTORY = PROJECT_ROOT / "data" / "sample"


@dataclass(frozen=True)
class AnswerEvaluationCase:
    question: str
    answer: str
    expected_concepts: list[str]


def calculate_concept_coverage(
    answer: str,
    expected_concepts: list[str],
) -> float:
    if not expected_concepts:
        return 0.0

    normalized_answer = answer.lower()

    matched = sum(
        1
        for concept in expected_concepts
        if concept.lower() in normalized_answer
    )

    return round(matched / len(expected_concepts), 4)


def create_dataset() -> list[AnswerEvaluationCase]:
    return [
        AnswerEvaluationCase(
            question="How many vacation days do employees receive?",
            answer=(
                "Employees receive 15 days of paid vacation per "
                "calendar year. "
                "[Document: employee_handbook, Page: 1]"
            ),
            expected_concepts=["15 days", "paid vacation"],
        ),
        AnswerEvaluationCase(
            question="How do employees enroll in the retirement plan?",
            answer=(
                "Employees can elect a contribution percentage "
                "through the benefits enrollment system. "
                "[Document: employee_benefits, Page: 1]"
            ),
            expected_concepts=[
                "contribution percentage",
                "benefits enrollment system",
            ],
        ),
        AnswerEvaluationCase(
            question="What are the password requirements?",
            answer=(
                "Passwords must contain at least 12 characters "
                "and include letters, numbers, and symbols. "
                "[Document: it_security_policy, Page: 1]"
            ),
            expected_concepts=[
                "12 characters",
                "letters",
                "numbers",
                "symbols",
            ],
        ),
        AnswerEvaluationCase(
            question="Is multi-factor authentication required?",
            answer=(
                "Multi-factor authentication is required for "
                "company email, cloud applications, and remote systems. "
                "[Document: it_security_policy, Page: 1]"
            ),
            expected_concepts=[
                "multi-factor authentication",
                "company email",
                "cloud applications",
                "remote systems",
            ],
        ),
        AnswerEvaluationCase(
            question="What should employees do when they are sick?",
            answer=(
                "Employees must notify their manager before "
                "their scheduled workday begins whenever possible. "
                "[Document: leave_attendance_policy, Page: 1]"
            ),
            expected_concepts=[
                "notify their manager",
                "scheduled workday begins",
                "whenever possible",
            ],
        ),
    ]


def find_evidence(
    question: str,
    documents,
) -> list[RerankedResult]:
    """Retrieve actual document chunks using BM25."""

    all_chunks = [
        chunk
        for document in documents
        for chunk in document.chunks
    ]

    retriever = BM25Retriever(all_chunks)
    results = retriever.search(question, top_k=5)

    return [
        RerankedResult(
            result=RetrievalResult(
                chunk=item.chunk,
                score=item.score,
            ),
            score=item.score,
        )
        for item in results
    ]


def main() -> None:
    print("\nLoading enterprise PDF documents...")

    pipeline = DocumentIngestionPipeline()
    documents = pipeline.process_directory(SAMPLE_DIRECTORY)

    print(f"Documents loaded: {len(documents)}")

    print("\nLoading semantic NLI evaluator...")
    semantic_evaluator = SemanticFaithfulnessEvaluator()

    print("\nLoading semantic embedding model...")
    semantic_embedder = SemanticEmbedder()

    evaluator = AnswerQualityEvaluator(
        semantic_evaluator=semantic_evaluator,
        semantic_embedder=semantic_embedder,
    )

    dataset = create_dataset()
    results = []

    for case in dataset:
        evidence = find_evidence(case.question, documents)

        if not evidence:
            print(f"WARNING: No evidence found for: {case.question}")
            continue

        result = evaluator.evaluate(
            question=case.question,
            answer=case.answer,
            evidence=evidence,
        )

        concept_coverage = calculate_concept_coverage(
            case.answer,
            case.expected_concepts,
        )

        retrieved_sources = sorted({
            (
                item.result.chunk.document_id,
                item.result.chunk.page_number,
            )
            for item in evidence
        })

        results.append({
            "question": case.question,
            "faithfulness": result.faithfulness_score,
            "relevance": result.answer_relevance_score,
            "semantic_faithfulness": result.semantic_faithfulness_score,
            "semantic_relevance": result.semantic_relevance_score,
            "concept_coverage": concept_coverage,
            "citation_accuracy": result.citation_accuracy_score,
            "retrieved_sources": retrieved_sources,
        })

    if not results:
        raise RuntimeError("No evaluation cases were completed.")

    count = len(results)

    print("\nREAL DOCUMENT ANSWER QUALITY EVALUATION")
    print("=" * 155)

    print(f"\nDocuments: {len(documents)}")
    print(f"Evaluation cases completed: {count}/{len(dataset)}\n")

    print(
        f"{'Question':<38}"
        f"{'Lexical':>10}"
        f"{'Relevant':>10}"
        f"{'Sem.Rel.':>10}"
        f"{'Sem.Faith':>10}"
        f"{'Concepts':>10}"
        f"{'Citation':>10}"
        f"{'Retrieved sources':>57}"
    )

    print("-" * 155)

    for item in results:
        question = item["question"][:35]

        sources = ", ".join(
            f"{document} (p.{page})"
            for document, page in item["retrieved_sources"]
        )

        semantic_relevance = item["semantic_relevance"]
        semantic_faithfulness = item["semantic_faithfulness"]

        print(
            f"{question:<38}"
            f"{item['faithfulness']:>10.4f}"
            f"{item['relevance']:>10.4f}"
            f"{semantic_relevance:>10.4f}"
            f"{semantic_faithfulness:>10.4f}"
            f"{item['concept_coverage']:>10.4f}"
            f"{item['citation_accuracy']:>10.4f}"
            f"{sources[:57]:>57}"
        )

    print("-" * 155)

    metrics = (
        "faithfulness",
        "relevance",
        "semantic_relevance",
        "semantic_faithfulness",
        "concept_coverage",
        "citation_accuracy",
    )

    print("\nAVERAGE EVALUATION METRICS")

    for metric in metrics:
        average = sum(
            item[metric] for item in results
        ) / count

        print(f"Average {metric}: {average:.4f}")

    print("=" * 155)
    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()
