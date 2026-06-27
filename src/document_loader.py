"""Load text from uploaded PDF and image documents."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from PIL import Image
from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class LoadedDocument:
    filename: str
    text: str
    page_count: int
    source_type: str


def load_uploaded_document(uploaded_file: BinaryIO, filename: str) -> LoadedDocument:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix}. Upload a PDF, JPG, JPEG, or PNG.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = Path(tmp.name)

    try:
        if suffix == ".pdf":
            return _load_pdf(tmp_path, filename)
        return _load_image(tmp_path, filename)
    finally:
        tmp_path.unlink(missing_ok=True)


def _load_pdf(path: Path, filename: str) -> LoadedDocument:
    reader = PdfReader(str(path))
    page_texts = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            page_texts.append(f"[Page {index}]\n{text.strip()}")

    combined = "\n\n".join(page_texts).strip()
    if not combined:
        raise ValueError("No selectable text found in the PDF. Try an image upload with OCR or a text-based PDF.")

    return LoadedDocument(
        filename=filename,
        text=combined,
        page_count=len(reader.pages),
        source_type="pdf",
    )


def _load_image(path: Path, filename: str) -> LoadedDocument:
    try:
        import pytesseract
    except ImportError as exc:
        raise RuntimeError("pytesseract is required for image OCR. Install requirements.txt first.") from exc

    image = Image.open(path)
    text = pytesseract.image_to_string(image).strip()
    if not text:
        raise ValueError("No text was detected in the image. Try a clearer scan or photo.")

    return LoadedDocument(
        filename=filename,
        text=text,
        page_count=1,
        source_type="image",
    )
