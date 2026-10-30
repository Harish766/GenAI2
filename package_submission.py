"""Package only audited evidence and final deliverables for ICT4403 Assignment 3."""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from finalize_results import ROOT, check_backup


OUTPUT = ROOT / "Harish_Bharathwaj_230953594_ICT4403_Assignment3_Submission.zip"
FILES = [
    "README.md", "experiment.py", "requirements.txt", "colab_runner.ipynb",
    "build_colab_notebook.py", "build_cover.py", "render_cover_pdf.py",
    "make_bundle.py", "finalize_results.py", "build_final_documents.py",
    "package_submission.py", ".gitignore",
    "tests/test_blocked_access.py", "tests/test_quality_and_isolation.py",
    "tests/test_submission_evidence.py",
    "Harish_Bharathwaj_230953594_FGAI_Assignment3.docx",
    "Harish_Bharathwaj_230953594_FGAI_Assignment3.pdf",
    "Harish_Bharathwaj_230953594_FGAI_Assignment3_Report.docx",
    "Harish_Bharathwaj_230953594_FGAI_Assignment3_Report.pdf",
    "results/REPORT.md", "results/run_status.json", "results/mt5_attempt_run_status.json",
    "results/validation.json", "results/comparison.png", "results/training_curves.png",
    "results/basic_history.json", "results/mt5_history.json",
]
for model in ("basic", "llama", "gemma"):
    FILES.extend(f"results/{model}_{suffix}" for suffix in
                 ("metrics.json", "translations.jsonl", "temperature.json"))


def build():
    check_backup()  # fail closed if evidence no longer matches the source archive
    missing = [name for name in FILES if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Submission files missing: {missing}")
    if (ROOT / "results" / "mt5_metrics.json").exists() or (ROOT / "results" / "mt5_translations.jsonl").exists():
        raise ValueError("Unexpected mT5 score or translation artifact must not enter this submission")
    with ZipFile(OUTPUT, "w", ZIP_DEFLATED) as archive:
        for name in FILES:
            archive.write(ROOT / name, arcname=name)
    with ZipFile(OUTPUT) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != set(FILES):
            raise ValueError("Submission ZIP failed integrity or manifest verification")
    return OUTPUT


if __name__ == "__main__":
    print(build())
