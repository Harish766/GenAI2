"""Build the clean three-cell Colab runner for the assignment."""
import json
from pathlib import Path


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True)}


notebook = {
    "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": [
            "# ICT4403 Assignment 3 - English to French comparison\n",
            "Select **Runtime > Change runtime type > T4 GPU**. Run the three code cells in order. Upload the current `assignment_project.zip` in cell 1. Cell 3 runs all four requested systems, records inaccessible or failed models honestly, then downloads the actual results. No new token is required for Basic Transformer or mT5-small. If Llama/Gemma approval later arrives, an existing token can be placed in Colab Secrets as `HF_TOKEN` without putting it in this notebook.\n",
        ]},
        code("""from google.colab import files
from datetime import datetime, timezone
import io, os, pathlib, zipfile
uploaded = files.upload()
assert 'assignment_project.zip' in uploaded, 'Upload the current assignment_project.zip'
workdir = pathlib.Path('/content/assignment_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
workdir.mkdir(parents=True)
with zipfile.ZipFile(io.BytesIO(uploaded['assignment_project.zip'])) as archive:
    assert archive.testzip() is None, 'Project ZIP is corrupt'
    archive.extractall(workdir)
os.chdir(workdir)
print('Project directory:', workdir)
"""),
        code("""%pip install -q -r requirements.txt
"""),
        code("""import json, os, pathlib, subprocess, sys, zipfile
from datetime import datetime, timezone
from huggingface_hub import get_token
try:
    from google.colab import userdata
    saved_token = userdata.get('HF_TOKEN')
except Exception:
    saved_token = None
if saved_token:
    os.environ['HF_TOKEN'] = saved_token
print('Existing Hugging Face token available:' , bool(get_token()))
subprocess.run([sys.executable, 'experiment.py'], check=True)
status_path = pathlib.Path('results/run_status.json')
status = json.loads(status_path.read_text(encoding='utf-8'))
assert status['test_size'] == 200, 'The assignment requires the 200-item test split'
current = set(status['current_run_files'])
assert 'results/REPORT.md' in current, 'Current-run report is missing'
for name in ('basic', 'mt5', 'llama', 'gemma'):
    entry = status['models'][name]
    print(name, entry['status'], entry['reason'])
    if entry['status'] == 'completed':
        metric_path = f'results/{name}_metrics.json'
        translation_path = f'results/{name}_translations.jsonl'
        assert metric_path in current and translation_path in current
        metric = json.loads(pathlib.Path(metric_path).read_text(encoding='utf-8'))
        assert metric['test_size'] == 200
        assert len(pathlib.Path(translation_path).read_text(encoding='utf-8').splitlines()) == 200
        print(f"  BLEU {metric['bleu']:.2f}; chrF {metric['chrf']:.2f}; empty {metric['empty_translations']}")
files_to_pack = [pathlib.Path(p) for p in sorted(current)]
files_to_pack += [status_path, pathlib.Path('experiment.py'), pathlib.Path('README.md')]
manifest = pathlib.Path('data/manifest.json')
if manifest.exists():
    files_to_pack.append(manifest)
assert all(p.is_file() for p in files_to_pack), 'A listed current-run artifact is missing'
archive_name = 'assignment_results_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.zip'
with zipfile.ZipFile(archive_name, 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in files_to_pack:
        archive.write(path)
print('Saved:', archive_name)
files.download(archive_name)
"""),
    ],
    "metadata": {"accelerator": "GPU", "colab": {"name": "ICT4403 Assignment 3 Runner"},
                 "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    "nbformat": 4, "nbformat_minor": 5,
}
Path("colab_runner.ipynb").write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
                                         encoding="utf-8")
print(Path("colab_runner.ipynb").resolve())
