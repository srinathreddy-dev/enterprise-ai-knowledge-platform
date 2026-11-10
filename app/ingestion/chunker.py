
from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    """A searchable piece of a document."""

    text: str
    page_number: int
    chunk_index: int
    document_id: str = "document"

    @property
    def chunk_id(self) -> str:
        """Return a stable identifier for this document chunk."""
        return (
            f"{self.document_id}-"
            f"page-{self.page_number}-"
            f"chunk-{self.chunk_index}"
        )


class TextChunker:
    """Split page text into overlapping, word-boundary-aware chunks."""

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 50,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")

        if overlap < 0:
            raise ValueError("overlap cannot be negative")

        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def _find_chunk_end(self, text: str, start: int) -> int:
        """Find a natural boundary near the maximum chunk size."""

        target = min(start + self.chunk_size, len(text))

        if target == len(text):
            return target

        minimum = start + max(1, self.chunk_size // 2)

        # Prefer sentence boundaries.
        for separator in (". ", "? ", "! ", "\n\n", "\n"):
            boundary = text.rfind(separator, minimum, target)

            if boundary != -1:
                return boundary + len(separator)

        # Otherwise prefer word boundaries.
        boundary = text.rfind(" ", minimum, target)

        if boundary != -1:
            return boundary + 1

        # If no boundary exists, use the hard limit.
        # This handles long unbroken strings safely.

        return target

    def chunk(
        self,
        text: str,
        page_number: int,
        document_id: str = "document",
    ) -> list[TextChunk]:
        """Split page text while preserving metadata and word boundaries."""

        text = text.strip()

        if not text:
            return []

        chunks: list[TextChunk] = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = self._find_chunk_end(text, start)
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    TextChunk(
                        text=chunk_text,
                        page_number=page_number,
                        chunk_index=chunk_index,
                        document_id=document_id,
                    )
                )

            if end >= len(text):
                break

            next_start = max(start + 1, end - self.overlap)

            # Move to the next word boundary when one exists
            # before the current chunk ends.
            boundary = text.find(" ", next_start, end)

            if boundary != -1:
                next_start = boundary + 1

            start = next_start
            chunk_index += 1

        return chunks

