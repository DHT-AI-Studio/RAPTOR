#!/usr/bin/env python3
"""Generate all test fixture files. Run once: python tests/fixtures/make_fixtures.py"""
import pathlib

OUT = pathlib.Path(__file__).parent


def make_scanned_pdf():
    """Create a minimal scanned-style PDF (image-only, no text layer)."""
    try:
        import io

        import fitz
        from PIL import Image, ImageDraw
    except ImportError:
        print("pip install Pillow pymupdf  to generate scanned.pdf")
        return
    img = Image.new("RGB", (595, 842), "white")
    draw = ImageDraw.Draw(img)
    draw.text((50, 50), "Scanned Document\nThis is a test scanned PDF.\nPage 1 of 1.", fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_image(page.rect, stream=buf.read())
    doc.save(str(OUT / "scanned.pdf"))
    print("Created scanned.pdf")


def make_docx():
    try:
        from docx import Document
    except ImportError:
        print("pip install python-docx  to generate sample.docx")
        return
    doc = Document()
    doc.add_heading("Sample DOCX Document", 0)
    doc.add_paragraph("This is a test DOCX file for DocAgent testing.")
    doc.add_heading("Section 1: Financial Data", level=1)
    doc.add_paragraph("Revenue: $1,000,000\nExpenses: $750,000\nProfit: $250,000")
    table = doc.add_table(rows=3, cols=3)
    table.style = "Table Grid"
    headers = ["Quarter", "Revenue", "Profit"]
    for i, h in enumerate(headers):
        table.rows[0].cells[i].text = h
    table.rows[1].cells[0].text = "Q1"; table.rows[1].cells[1].text = "$250,000"; table.rows[1].cells[2].text = "$60,000"
    table.rows[2].cells[0].text = "Q2"; table.rows[2].cells[1].text = "$300,000"; table.rows[2].cells[2].text = "$80,000"
    doc.save(str(OUT / "sample.docx"))
    print("Created sample.docx")


def make_xlsx():
    try:
        import openpyxl
    except ImportError:
        print("pip install openpyxl  to generate sample.xlsx")
        return
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Financial Report"
    ws.append(["Quarter", "Revenue", "Expenses", "Profit"])
    ws.append(["Q1 2026", 250000, 190000, 60000])
    ws.append(["Q2 2026", 300000, 220000, 80000])
    ws.append(["Q3 2026", 280000, 200000, 80000])
    ws.append(["Q4 2026", 320000, 230000, 90000])
    wb.save(str(OUT / "sample.xlsx"))
    print("Created sample.xlsx")


def make_csv():
    content = "Quarter,Revenue,Expenses,Profit\nQ1 2026,250000,190000,60000\nQ2 2026,300000,220000,80000\n"
    (OUT / "sample.csv").write_text(content)
    print("Created sample.csv")


def make_txt():
    content = (
        "DocAgent Test Document\n\n"
        "This document is used for testing the DocAgent service.\n"
        "It contains plain text content for classify, extract, summarize, qa, and search tasks.\n\n"
        "Key Information:\n"
        "- Project: Raptor Platform\n"
        "- Version: 1.0\n"
        "- Date: 2026-10-16\n"
        "- Author: Test Suite\n"
    )
    (OUT / "sample.txt").write_text(content)
    print("Created sample.txt")


def make_html():
    content = """<!DOCTYPE html>
<html><head><title>Sample HTML</title></head>
<body>
<h1>DocAgent HTML Test</h1>
<p>This is a test HTML document for the DocAgent service.</p>
<table border="1">
  <tr><th>Item</th><th>Value</th></tr>
  <tr><td>Revenue</td><td>$1,000,000</td></tr>
  <tr><td>Profit</td><td>$250,000</td></tr>
</table>
</body></html>"""
    (OUT / "sample.html").write_text(content)
    print("Created sample.html")


def make_png():
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("pip install Pillow  to generate sample.png")
        return
    img = Image.new("RGB", (400, 300), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 380, 280], outline="black", width=2)
    draw.text((40, 40), "DocAgent Test Image\n\nThis PNG contains text\nfor VLM testing.", fill="black")
    img.save(str(OUT / "sample.png"))
    print("Created sample.png")


if __name__ == "__main__":
    make_scanned_pdf()
    make_docx()
    make_xlsx()
    make_csv()
    make_txt()
    make_html()
    make_png()
    print("\nAll fixtures generated in", OUT)
