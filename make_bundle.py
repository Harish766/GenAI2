"""Pack only the reviewed source files for a clean Colab upload."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

files = [
    Path("experiment.py"), Path("requirements.txt"), Path("README.md"),
    Path(".gitignore"), Path("build_cover.py"), Path("render_cover_pdf.py"),
    Path("build_colab_notebook.py"), Path("make_bundle.py"),
    Path("colab_runner.ipynb"),
    Path("tests/test_blocked_access.py"), Path("tests/test_quality_and_isolation.py"),
    Path("Harish_Bharathwaj_230953594_FGAI_Assignment3.docx"),
    Path("Harish_Bharathwaj_230953594_FGAI_Assignment3.pdf"),
]
missing = [str(path) for path in files if not path.is_file()]
if missing:
    raise FileNotFoundError(f"Required project files missing: {missing}")
with ZipFile("assignment_project.zip", "w", ZIP_DEFLATED) as bundle:
    for path in files:
        bundle.write(path)
print(Path("assignment_project.zip").resolve())
