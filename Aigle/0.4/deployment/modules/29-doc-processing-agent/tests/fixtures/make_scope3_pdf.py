"""Generate the 3-page Scope 3 emissions PDF fixture used by DA-6 tests.

Run once (needs reportlab):  python tests/fixtures/make_scope3_pdf.py
Produces: tests/fixtures/scope3_emissions.pdf
"""
from __future__ import annotations

import os

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas

PAGES = [
    (
        "Scope 3 Greenhouse Gas Emissions — Overview",
        [
            "Scope 3 emissions are all indirect greenhouse gas emissions that occur "
            "in a company's value chain, excluding Scope 1 and Scope 2.",
            "For most organisations, Scope 3 supplier emissions dominate the total "
            "carbon footprint, often exceeding 70 percent of company-wide emissions.",
            "The GHG Protocol defines fifteen Scope 3 categories, split between "
            "upstream and downstream activities across the supply chain.",
            "Purchased goods and services (Category 1) captures cradle-to-gate "
            "emissions from suppliers and is typically the single largest source.",
        ],
    ),
    (
        "Scope 3 Supplier Emissions",
        [
            "Scope 3 supplier emissions are the greenhouse gas emissions produced "
            "by suppliers across the upstream value chain.",
            "Supplier-specific data is the highest-quality method for quantifying "
            "Scope 3 supplier emissions, using primary activity data from vendors.",
            "Where primary data is unavailable, spend-based estimation applies "
            "emission factors to procurement spend per supplier category.",
            "A supplier engagement programme collects product carbon footprints "
            "from strategic suppliers to replace industry-average factors.",
            "Data quality improves as more suppliers report verified emissions "
            "aligned with the Science Based Targets initiative.",
        ],
    ),
    (
        "Reducing Value-Chain Emissions",
        [
            "Reduction levers include supplier selection based on carbon intensity, "
            "low-carbon material substitution, and logistics optimisation.",
            "Setting supplier emission reduction targets and tracking progress is "
            "central to a credible Scope 3 decarbonisation strategy.",
            "Category 4 (upstream transportation) and Category 11 (use of sold "
            "products) are frequently material downstream hotspots.",
            "Transparent Scope 3 reporting builds trust with investors and "
            "regulators demanding value-chain climate disclosure.",
        ],
    ),
]


def build(path: str) -> None:
    c = canvas.Canvas(path, pagesize=A4)
    width, height = A4
    for title, paragraphs in PAGES:
        c.setFont("Helvetica-Bold", 16)
        c.drawString(2 * cm, height - 3 * cm, title)
        c.setFont("Helvetica", 11)
        y = height - 4.5 * cm
        for para in paragraphs:
            # naive wrap at ~90 chars
            line = ""
            for word in para.split():
                if len(line) + len(word) + 1 > 90:
                    c.drawString(2 * cm, y, line)
                    y -= 0.7 * cm
                    line = word
                else:
                    line = f"{line} {word}".strip()
            if line:
                c.drawString(2 * cm, y, line)
                y -= 1.1 * cm
        c.showPage()
    c.save()


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scope3_emissions.pdf")
    build(out)
    print(f"wrote {out}")
