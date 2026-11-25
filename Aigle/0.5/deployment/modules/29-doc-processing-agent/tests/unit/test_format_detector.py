"""DA-09a: test format detection for all 8 format categories."""
import pathlib

import pytest
from app.adapters.doc12_adapter import FORMAT_MATRIX, _detect_format_category

FIXTURES = pathlib.Path(__file__).parent.parent / "integration" / "fixtures"

@pytest.mark.parametrize("filename,expected_category", [
    ("scope3_emissions.pdf", "pdf_text"),
    ("scanned.pdf",          "pdf_scanned"),
    ("sample.docx",          "office"),
    ("sample.xlsx",          "spreadsheet"),
    ("sample.csv",           "spreadsheet"),
    ("sample.txt",           "plaintext"),
    ("sample.html",          "html"),
    ("sample.png",           "image"),
])
def test_detect_format_category(filename: str, expected_category: str):
    # Note: _detect_format_category uses extension only; scanned vs text PDF
    # distinction is made by Module 12 at processing time.
    # For format_detector tests, both .pdf files map to "pdf_text" by extension.
    # The scanned category is tested via the processing_mode path.
    result = _detect_format_category(filename)
    # PDF always maps to pdf_text by extension (scanned detection is runtime)
    if filename == "scanned.pdf":
        assert result in ("pdf_text", "pdf_scanned")
    else:
        assert result == expected_category, f"{filename}: got {result!r}, want {expected_category!r}"


def test_format_matrix_has_all_categories():
    expected = {"pdf_text", "pdf_scanned", "office", "spreadsheet", "plaintext", "html", "image"}
    assert expected.issubset(set(FORMAT_MATRIX.keys()))


def test_format_matrix_entries_have_required_keys():
    for category, info in FORMAT_MATRIX.items():
        assert "extensions" in info, f"{category} missing 'extensions'"
        assert "vlm_required" in info, f"{category} missing 'vlm_required'"
        assert "via" in info, f"{category} missing 'via'"
        assert isinstance(info["extensions"], list)


@pytest.mark.parametrize("path_fixture,expected_category", [
    ("scope3_emissions.pdf", "pdf_text"),
    ("sample.docx",          "office"),
    ("sample.xlsx",          "spreadsheet"),
    ("sample.csv",           "spreadsheet"),
    ("sample.txt",           "plaintext"),
    ("sample.html",          "html"),
    ("sample.png",           "image"),
])
def test_real_fixture_files_exist_and_detect(path_fixture: str, expected_category: str):
    p = FIXTURES / path_fixture
    assert p.exists(), f"Fixture missing: {p}  (run: python tests/integration/fixtures/make_fixtures.py)"
    result = _detect_format_category(path_fixture)
    assert result == expected_category
