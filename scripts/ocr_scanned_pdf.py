"""Transcribe a scanned corpus PDF into a traceable Markdown sidecar.

Usage: python scripts/ocr_scanned_pdf.py data/file.pdf data/file.ocr.md --cache-dir path
Requires OPENAI_API_KEY and Poppler's pdftoppm. Pages are cached for resumability.
Review numbers against the original scan before using them for financial decisions.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import os
from pathlib import Path
import shutil
import subprocess

from pypdf import PdfReader


def transcribe(pdf: Path, output: Path, cache_dir: Path, model: str, renderer: str) -> None:
    from openai import OpenAI

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for OCR")
    binary = shutil.which(renderer) if not Path(renderer).is_file() else renderer
    if not binary:
        raise FileNotFoundError(f"pdftoppm not found: {renderer}")
    digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
    pages = len(PdfReader(str(pdf)).pages)
    cache_dir = cache_dir / digest[:16]
    cache_dir.mkdir(parents=True, exist_ok=True)
    client = OpenAI()
    blocks = []
    for page in range(1, pages + 1):
        text_path = cache_dir / f"page-{page:03d}.md"
        if not text_path.exists():
            prefix = cache_dir / f"page-{page:03d}"
            subprocess.run([str(binary), "-f", str(page), "-l", str(page),
                            "-scale-to", "1800", "-png", "-singlefile",
                            str(pdf), str(prefix)], check=True, stdout=subprocess.DEVNULL)
            image_path = prefix.with_suffix(".png")
            image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content":
                     "Bạn làm OCR tài liệu tiếng Việt. Chép lại chính xác chữ và số nhìn thấy. "
                     "Giữ tiêu đề, điều khoản, thứ tự hàng/cột và bảng ở dạng Markdown. "
                     "Không tóm tắt, không suy đoán hay tự thêm nội dung. "
                     "Nếu một từ hoặc số không đọc rõ, ghi [không rõ]. "
                     "Chỉ xuất văn bản OCR, không bọc trong code fence."},
                    {"role": "user", "content": [
                        {"type": "text", "text": f"Trang {page}/{pages} của {pdf.name}. Hãy OCR toàn trang."},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{image_data}", "detail": "high"}},
                    ]},
                ],
            )
            content = response.choices[0].message.content
            if not isinstance(content, str) or len(content.strip()) < 20:
                raise RuntimeError(f"OCR page {page} returned too little text")
            content = content.strip()
            if "can't assist with that" in content.lower() or "cannot assist with that" in content.lower():
                raise RuntimeError(f"OCR page {page} was refused; no text has been saved")
            if content.startswith("```") and content.endswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            text_path.write_text(content.strip() + "\n", encoding="utf-8")
            image_path.unlink(missing_ok=True)
        page_text = text_path.read_text(encoding="utf-8").strip()
        if "can't assist with that" in page_text.lower() or "cannot assist with that" in page_text.lower():
            raise RuntimeError(f"Cached OCR page {page} is a refusal; delete it and retry")
        blocks.append(f"## Trang {page}\n\n{page_text}")
        print(f"{pdf.name}: {page}/{pages} pages", flush=True)

    header = (f"# OCR của {pdf.name}\n\n"
              f"Nguồn: `{pdf.name}`; SHA-256 PDF: `{digest}`. "
              "Đây là bản chép máy để tìm kiếm; khi dùng số liệu hoặc điều khoản, "
              "đối chiếu bản PDF gốc.\n\n")
    output.write_text(header + "\n\n".join(blocks) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--pdftoppm", default="pdftoppm")
    args = parser.parse_args()
    transcribe(args.pdf, args.output, args.cache_dir, args.model, args.pdftoppm)
