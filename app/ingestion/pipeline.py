from pathlib import Path

from app.ingestion.pdf_loader import LoadedDocument, PDFLoader
from app.ingestion.text_cleaner import TextCleaner


class DocumentIngestionPipeline:
    """Coordinate document loading and text normalization."""

    def __init__(
        self,
        pdf_loader: PDFLoader | None = None,
        text_cleaner: TextCleaner | None = None,
    ) -> None:
        self.pdf_loader = pdf_loader or PDFLoader()
        self.text_cleaner = text_cleaner or TextCleaner()

    def process(self, file_path: str | Path) -> LoadedDocument:
        """Load a document and normalize the text on each page."""
        document = self.pdf_loader.load(file_path)

        cleaned_pages = [
            page.__class__(
                page_number=page.page_number,
                text=self.text_cleaner.clean(page.text),
            )
            for page in document.pages
        ]

        return LoadedDocument(
            file_name=document.file_name,
            pages=cleaned_pages,
        )
