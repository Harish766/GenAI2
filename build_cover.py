"""Create the required assignment cover document."""
from __future__ import annotations

import argparse
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor


def build(repo_url: str):
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(0.9)
    section.bottom_margin = Inches(0.8)
    section.left_margin = Inches(0.9)
    section.right_margin = Inches(0.9)
    styles = doc.styles
    styles["Normal"].font.name = "Aptos"
    styles["Normal"].font.size = Pt(11)
    styles["Normal"].font.color.rgb = RGBColor(30, 41, 59)
    styles["Title"].font.name = "Aptos Display"
    styles["Title"].font.size = Pt(21)
    styles["Title"].font.bold = True
    styles["Title"].font.color.rgb = RGBColor(0, 0, 0)

    top = doc.add_paragraph()
    top.style = "Title"
    top.alignment = WD_ALIGN_PARAGRAPH.CENTER
    top.add_run("English to French Machine Translation Comparison")
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run("Foundations of Generative AI  ICT4403  Assignment 3")
    run.bold = True
    run.font.size = Pt(11)
    subtitle.space_after = Pt(24)

    intro = doc.add_paragraph()
    intro.add_run("Submission summary. ").bold = True
    intro.add_run("This project compares a basic Transformer, mT5, Llama, and Gemma for English to French translation using a shared held out test set.")
    intro.space_after = Pt(18)

    table = doc.add_table(rows=0, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Light Shading Accent 1"
    fields = [
        ("Student name", "Harish Bharathwaj"),
        ("Registration number", "230953594"),
        ("Assignment title", "Foundations of Generative AI Assignment 3"),
        ("System title", "English to French Autoregressive Translation Systems"),
        ("Dataset", "OPUS Books English French parallel text"),
        ("Dataset link", "https://huggingface.co/datasets/Helsinki-NLP/opus_books"),
        ("GitHub repository", repo_url),
    ]
    for label, value in fields:
        cells = table.add_row().cells
        for cell in cells:
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        cells[0].text = label
        cells[1].text = value
        for run in cells[0].paragraphs[0].runs:
            run.bold = True
    table.columns[0].width = Inches(1.65)
    table.columns[1].width = Inches(4.8)

    note = doc.add_paragraph()
    note.space_before = Pt(18)
    note.add_run("Repository contents. ").bold = True
    note.add_run("The repository includes the implementation and execution steps. Measured results, generated translations, and the comparison report are produced when the experiment runs.")
    output = Path("Harish_Bharathwaj_230953594_FGAI_Assignment3.docx")
    doc.save(output)
    print(output.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-url", default="Repository link pending publication")
    build(parser.parse_args().repo_url)
