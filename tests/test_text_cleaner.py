from app.ingestion.text_cleaner import TextCleaner


def test_cleaner_normalizes_spaces_and_line_breaks():
    cleaner = TextCleaner()

    text = (
        "Employee    Benefits\r\n"
        "\r\n"
        "\r\n"
        "Employees are eligible for health insurance."
    )

    cleaned_text = cleaner.clean(text)

    assert cleaned_text == (
        "Employee Benefits\n\n"
        "Employees are eligible for health insurance."
    )


def test_cleaner_removes_leading_and_trailing_whitespace():
    cleaner = TextCleaner()

    text = "   Employee handbook information   "

    cleaned_text = cleaner.clean(text)

    assert cleaned_text == "Employee handbook information"


def test_cleaner_preserves_meaningful_single_line_breaks():
    cleaner = TextCleaner()

    text = "Eligibility requirements:\nFull-time employees only."

    cleaned_text = cleaner.clean(text)

    assert cleaned_text == (
        "Eligibility requirements:\n"
        "Full-time employees only."
    )
