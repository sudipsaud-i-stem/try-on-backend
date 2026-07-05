from __future__ import annotations


class MaskValidationError(Exception):
    """Raised when garment mask fails validity checks and no fallback succeeded."""

    def __init__(self, message: str, code: str = "mask_invalid", diagnostics: dict | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.diagnostics = diagnostics or {}
