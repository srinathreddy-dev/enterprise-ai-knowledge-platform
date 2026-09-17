from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    """A searchable piece of a document."""

    text: str
    page_number: int
    chunk_index: int


class TextChunker:
    """Split page text into smaller overlapping chunks."""

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")

        if overlap < 0:
            raise ValueError("overlap cannot be negative")

        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, page_number: int) -> list[TextChunk]:
        """Split text into chunks while preserving page metadata."""
        text = text.strip()

        if not text:
            return []

        chunks: list[TextChunk] = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    TextChunk(
                        text=chunk_text,
                        page_number=page_number,
                        chunk_index=chunk_index,
                    )
                )

            if end == len(text):
                break

            start = end - self.overlap
            chunk_index += 1

        return chunks
