"""
Layer 0 — pull text + page images out of a PDF (PyMuPDF / fitz).
Used as the fast baseline before Nougat/VLM.
"""

import fitz  # PyMuPDF
from pathlib import Path


class Layer0_PDFIngestion:
    """Stores paths for PDFs and for rendered page PNGs."""

    def __init__(self, pdf_dir: str | Path, img_dir: str | Path):
        self.pdf_dir = Path(pdf_dir)
        self.img_dir = Path(img_dir)

    def extract_metadata(self, pdf_path: str | Path) -> dict:
        """Lightweight metadata PyMuPDF exposes (pages, producer, size)."""
        pdf_path = Path(pdf_path)
        doc = fitz.open(str(pdf_path))
        meta = doc.metadata
        info = {
            "file": pdf_path.name,
            "pages": doc.page_count,
            "author": meta.get("author", ""),
            "creator": meta.get("creator", ""),
            "producer": meta.get("producer", ""),
            "file_size_kb": round(pdf_path.stat().st_size / 1024, 1),
        }
        doc.close()
        return info

    def extract_text(self, pdf_path: str | Path) -> list[dict]:
        """Plain `get_text` per page — good for native PDFs, weak on scans."""
        doc = fitz.open(str(pdf_path))
        pages = []
        for i, page in enumerate(doc):
            text = page.get_text("text")
            pages.append({
                "page": i + 1,
                "text": text,
                "char_count": len(text),
            })
        doc.close()
        return pages

    def extract_images(self, pdf_path: str | Path, dpi: int = 300) -> list[dict]:
        """Rasterize each page to PNG under `img_dir/<stem>/` for Nougat/VLM."""
        pdf_path = Path(pdf_path)
        doc = fitz.open(str(pdf_path))
        name = pdf_path.stem
        out_dir = self.img_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)

        images = []
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=dpi)
            img_path = out_dir / f"page_{i + 1}.png"
            pix.save(str(img_path))
            images.append({
                "page": i + 1,
                "path": str(img_path),
                "width": pix.width,
                "height": pix.height,
                "size_kb": round(img_path.stat().st_size / 1024, 1),
            })
        doc.close()
        return images

    def analyze_text_quality(self, pages: list[dict]) -> dict:
        """Cheap heuristics: did we lose integrals, superscripts, greek letters?"""
        full_text = " ".join(p["text"] for p in pages)

        checks = {
            "integral_symbol": "\u222c" in full_text or "\u222b" in full_text,
            "partial_deriv": "\u2202" in full_text,
            "sqrt_symbol": "\u221a" in full_text,
            "pi_symbol": "\u03c0" in full_text,
            "greek_letters": any(c in full_text for c in "\u03d5\u03b8\u03b1\u03b2\u03b3\u03b4"),
            "superscripts": any(c in full_text for c in "\u00b2\u00b3\u2074\u2075"),
            "no_garbled_ZZ": "ZZ" not in full_text,
        }

        issues = []
        if not checks["integral_symbol"]:
            issues.append("Integral symbol missing (often rendered as 'ZZ' or 'Z')")
        if not checks["superscripts"]:
            issues.append("Superscripts missing (x^2 flattened to x2)")
        if not checks["no_garbled_ZZ"]:
            issues.append("Double integral garbled as 'ZZ'")

        return {
            "checks": checks,
            "issues": issues,
            "score": sum(checks.values()),
            "max_score": len(checks),
            "total_chars": len(full_text),
        }

    def process(self, pdf_path: str | Path, verbose: bool = True) -> dict:
        """One-shot helper: metadata, text quality, and page images (verbose CLI)."""
        pdf_path = Path(pdf_path)
        fname = pdf_path.name

        if verbose:
            print(f"\n{'='*60}")
            print(f"  {fname}")
            print(f"{'='*60}")

        meta = self.extract_metadata(pdf_path)
        if verbose:
            print(f"\n  Metadata: {meta['pages']} pages, {meta['file_size_kb']} KB")
            print(f"  Creator: {meta['creator']}")

        pages = self.extract_text(pdf_path)
        quality = self.analyze_text_quality(pages)

        if verbose:
            print(f"\n  Baseline text extraction:")
            for pg in pages:
                preview = pg["text"][:120].replace("\n", " ").strip()
                print(f"    Page {pg['page']}: {pg['char_count']} chars")
                print(f"    Preview: '{preview}...'")

            print(f"\n  Quality score: {quality['score']}/{quality['max_score']}")
            for key, val in quality["checks"].items():
                icon = "[OK]" if val else "[FAIL]"
                print(f"    {icon} {key}")
            for issue in quality["issues"]:
                print(f"    [!] {issue}")

        images = self.extract_images(pdf_path)

        if verbose:
            print(f"\n  Page images:")
            for img in images:
                print(f"    Page {img['page']}: {img['width']}x{img['height']}px, {img['size_kb']} KB")

        return {
            "file": fname,
            "metadata": meta,
            "text_pages": pages,
            "quality": quality,
            "images": images,
        }
