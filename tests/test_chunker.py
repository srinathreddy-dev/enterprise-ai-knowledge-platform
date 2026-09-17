import pytest

from app.ingestion.chunker import TextChunker


def test_chunker_splits_long_text_with_overlap():
    text = "A" * 120

    chunker = TextChunker(chunk_size=50, overlap=10)

    chunks = chunker.chunk(text, page_number=2)

    assert len(chunks) == 3
    assert chunks[0].page_number == 2
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1
    assert chunks[2].chunk_index == 2
    assert len(chunks[0].text) == 50
    assert len(chunks[1].text) == 50
    assert len(chunks[2].text) == 40


def test_chunker_returns_empty_list_for_empty_text():
    chunker = TextChunker()

    chunks = chunker.chunk("   ", page_number=1)

    assert chunks == []


def test_chunker_preserves_short_text():
    chunker = TextChunker(chunk_size=100, overlap=10)

    chunks = chunker.chunk(
        "Employee benefits information.",
        page_number=3,
    )

    assert len(chunks) == 1
    assert chunks[0].text == "Employee benefits information."
    assert chunks[0].page_number == 3


def test_chunker_rejects_invalid_settings():
    with pytest.raises(ValueError, match="chunk_size"):
        TextChunker(chunk_size=0)

    with pytest.raises(ValueError, match="overlap cannot"):
        TextChunker(chunk_size=100, overlap=-1)

    with pytest.raises(ValueError, match="overlap must"):
        TextChunker(chunk_size=100, overlap=100)
