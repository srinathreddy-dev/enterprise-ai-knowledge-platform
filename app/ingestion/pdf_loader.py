from dataclasses import dataclass
from pathlib import Path

import pymupdf


@dataclass(frozen=True)
class PageContent:
    """Text extracted from a single PDF page."""

    page_number: int
    text: str


@dataclass(frozen=True)
class LoadedDocument:
    """Content extracted from a single source document."""

    file_name: str
    pages: list[PageContent]


class PDFLoader:
    """Load text content from a PDF document."""

    def load(self, file_path: str | Path) -> LoadedDocument:
        """Extract text from every page in the PDF."""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        if path.suffix.lower() != ".pdf":
            raise ValueError(
                f"Expected a PDF file, received: {path.suffix}"
            )

        pages: list[PageContent] = []

        with pymupdf.open(path) as document:
            for page_number, page in enumerate(document, start=1):
                text = page.get_text("text").strip()

                pages.append(
                    PageContent(
                        page_number=page_number,
                        text=text,
                    )
                )

        return LoadedDocument(
            file_name=path.name,
            pages=pages,
        )
