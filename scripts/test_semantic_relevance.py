
from app.retrieval.semantic_retriever import SemanticEmbedder


def main():
    print("\nLoading semantic embedding model...")
    embedder = SemanticEmbedder()

    test_cases = [
        {
            "question": "What are the password requirements?",
            "answer": (
                "Passwords must contain at least 12 characters "
                "and include letters, numbers, and symbols."
            ),
        },
        {
            "question": "What are the password requirements?",
            "answer": (
                "Employees receive 15 days of paid vacation "
                "per calendar year."
            ),
        },
        {
            "question": "How do employees enroll in the retirement plan?",
            "answer": (
                "Employees can select their contribution percentage "
                "through the benefits enrollment system."
            ),
        },
        {
            "question": "How do employees enroll in the retirement plan?",
            "answer": (
                "Employees must notify their manager before "
                "their scheduled workday begins."
            ),
        },
    ]

    print("\nSEMANTIC RELEVANCE VALIDATION")
    print("=" * 90)

    for case in test_cases:
        question_embedding = embedder.embed_query(case["question"])
        answer_embedding = embedder.embed_query(case["answer"])

        import numpy as np

        similarity = float(
            np.dot(question_embedding, answer_embedding)
        )

        print(f"\nQuestion: {case['question']}")
        print(f"Answer:   {case['answer']}")
        print(f"Similarity score: {similarity:.4f}")

    print("\n" + "=" * 90)
    print("Validation run completed.")


if __name__ == "__main__":
    main()
