
from dataclasses import dataclass
from pathlib import Path

from app.ingestion.chunker import TextChunk, TextChunker
from app.ingestion.pdf_loader import LoadedDocument, PDFLoader
from app.ingestion.text_cleaner import TextCleaner


@dataclass(frozen=True)
class IngestedDocument:
    """Document content after cleaning and chunking."""

    document: LoadedDocument
    chunks: list[TextChunk]


class DocumentIngestionPipeline:
    """Coordinate document loading, cleaning, and chunking."""

    def __init__(
        self,
        pdf_loader: PDFLoader | None = None,
        text_cleaner: TextCleaner | None = None,
        text_chunker: TextChunker | None = None,
    ) -> None:
        self.pdf_loader = pdf_loader or PDFLoader()
        self.text_cleaner = text_cleaner or TextCleaner()
        self.text_chunker = text_chunker or TextChunker()

    def process(self, file_path: str | Path) -> IngestedDocument:
        """Load, clean, and chunk a PDF document."""
        path = Path(file_path)

        document = self.pdf_loader.load(path)

        document_id = path.stem

        cleaned_pages = [
            page.__class__(
                page_number=page.page_number,
                text=self.text_cleaner.clean(page.text),
            )
            for page in document.pages
        ]

        cleaned_document = LoadedDocument(
            file_name=document.file_name,
            pages=cleaned_pages,
        )

        chunks: list[TextChunk] = []

        for page in cleaned_document.pages:
            chunks.extend(
                self.text_chunker.chunk(
                    page.text,
                    page.page_number,
                    document_id=document_id,
                )
            )

        return IngestedDocument(
            document=cleaned_document,
            chunks=chunks,
        )

