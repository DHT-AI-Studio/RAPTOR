"""P13: Module 12 reads when a document was written from the file itself (PDF /CreationDate, DOCX / PPTX
docProps/core.xml) and puts it on every chunk; Module 25 validates it and dates deictic phrases from it."""
import importlib.util
import os
import zipfile

# Loaded from its file, not via sys.path: the analysis service has its own config.py, which would shadow the
# orchestrator's for the other tests in this directory.
_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                     "worker", "services", "document_analysis_service", "source_dates.py")
_spec = importlib.util.spec_from_file_location("m12_source_dates", _PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
read_source_date = _mod.read_source_date

CORE = ('<?xml version="1.0" encoding="UTF-8"?><cp:coreProperties '
        'xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        '{created}</cp:coreProperties>')


def _pdf(path, created):
    import fitz
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Google released the phone two years ago.")
    doc.set_metadata({"creationDate": created} if created else {})
    doc.save(str(path))
    doc.close()


def _office(path, created_xml):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("docProps/core.xml", CORE.format(created=created_xml))
        z.writestr("word/document.xml", "<w:document/>")


def test_pdf_creation_date(tmp_path):
    _pdf(tmp_path / "f", "D:20150320121500+08'00'")
    assert read_source_date(tmp_path / "f", "Report.PDF") == ("D:20150320121500+08'00'", "pdf_creation_date")


def test_pdf_without_a_date(tmp_path):
    _pdf(tmp_path / "f", None)
    assert read_source_date(tmp_path / "f", "report.pdf") == (None, None)


def test_docx_and_pptx_core_created(tmp_path):
    _office(tmp_path / "d", '<dcterms:created xsi:type="dcterms:W3CDTF">2015-03-20T04:15:00Z</dcterms:created>')
    assert read_source_date(tmp_path / "d", "notes.docx") == ("2015-03-20T04:15:00Z", "docx_core_created")
    assert read_source_date(tmp_path / "d", "deck.pptx") == ("2015-03-20T04:15:00Z", "docx_core_created")


def test_docx_without_created(tmp_path):
    _office(tmp_path / "d", "")
    assert read_source_date(tmp_path / "d", "notes.docx") == (None, None)


def test_damaged_or_other_files_lose_only_their_date(tmp_path):
    (tmp_path / "bad").write_bytes(b"not a pdf at all")
    assert read_source_date(tmp_path / "bad", "broken.pdf") == (None, None)
    assert read_source_date(tmp_path / "bad", "broken.docx") == (None, None)
    assert read_source_date(tmp_path / "bad", "notes.txt") == (None, None)
