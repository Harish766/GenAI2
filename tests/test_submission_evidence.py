"""Verify the reconciled report comes only from the archived observed results."""
import json
import re
import sys
import unittest
from pathlib import Path
from zipfile import ZipFile

from docx import Document
from pypdf import PdfReader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from finalize_results import BACKUP, ROOT, check_backup


class SubmissionEvidenceTests(unittest.TestCase):
    def test_three_completed_models_share_the_archived_test_split(self):
        evidence = check_backup()
        self.assertEqual(set(evidence["metrics"]), {"basic", "llama", "gemma"})
        self.assertEqual(evidence["test_hash"], evidence["original_status"]["split_sha256"]["test"])
        for name in ("basic", "llama", "gemma"):
            self.assertEqual(len(evidence["translations"][name]), 200)
            self.assertEqual(evidence["metrics"][name]["test_size"], 200)
        self.assertEqual(evidence["source_copies"], {"basic": 0, "llama": 5, "gemma": 45})
        self.assertEqual(evidence["original_status"]["models"]["mt5"]["status"], "failed")

    def test_reconciled_report_and_status_have_no_mt5_score(self):
        report = (ROOT / "results" / "REPORT.md").read_text(encoding="utf-8")
        status = json.loads((ROOT / "results" / "run_status.json").read_text(encoding="utf-8"))
        self.assertIn("mT5-small with LoRA | Failed during generation | - | -", report)
        self.assertIn("Llama ranks first", report)
        self.assertIn("not independently recomputed", report)
        self.assertEqual(status["models"]["mt5"]["status"], "failed")
        self.assertIn("not_a_single_invocation", status["record_type"])
        self.assertFalse((ROOT / "results" / "mt5_metrics.json").exists())
        self.assertFalse((ROOT / "results" / "mt5_translations.jsonl").exists())
        self.assertTrue(BACKUP.is_file())

    def test_final_document_structure_and_identity(self):
        docx_path = ROOT / "Harish_Bharathwaj_230953594_FGAI_Assignment3_Report.docx"
        pdf_path = docx_path.with_suffix(".pdf")
        with ZipFile(docx_path) as archive:
            self.assertIsNone(archive.testzip())
        document = Document(docx_path)
        text = "\n".join(p.text for p in document.paragraphs)
        self.assertIn("Harish Bharathwaj", text)
        self.assertIn("230953594", text)
        self.assertIn("mT5 emitted a sentinel token", text)
        self.assertIn("Sampling temperature analysis", text)
        self.assertEqual(len(document.tables), 2)
        pdf = PdfReader(str(pdf_path))
        self.assertGreaterEqual(len(pdf.pages), 3)
        pdf_text = re.sub(r"\s+", " ", "\n".join(page.extract_text() for page in pdf.pages))
        self.assertIn("Harish Bharathwaj", pdf_text)
        self.assertIn("mT5 emitted a sentinel token", pdf_text)
        self.assertIn("45 of its 200 outputs", pdf_text)


if __name__ == "__main__":
    unittest.main()
