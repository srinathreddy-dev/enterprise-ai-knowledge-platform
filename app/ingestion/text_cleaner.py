import re


class TextCleaner:
    """Normalize extracted document text without changing its meaning."""

    def clean(self, text: str) -> str:
        """Return normalized text suitable for downstream processing."""
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        text = re.sub(r"[ \t]+", " ", text)

        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()
