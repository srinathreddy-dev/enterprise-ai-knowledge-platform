
import re


class TextCleaner:
    """Normalize extracted document text without changing its meaning."""

    def clean(self, text: str) -> str:
        """Return normalized text suitable for downstream processing."""

        # Normalize Windows and Mac line endings.
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Normalize Unicode line and paragraph separators.
        text = text.replace("\u2028", "\n")
        text = text.replace("\u2029", "\n")

        # Normalize non-breaking spaces.
        text = text.replace("\u00a0", " ")

        # Normalize repeated spaces and tabs.
        text = re.sub(r"[ \t]+", " ", text)

        # Normalize excessive blank lines.
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()
