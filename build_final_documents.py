"""Create matching evidence-based Word and PDF reports from audited Colab files."""
from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

from finalize_results import BACKUP, MODEL_LABELS, ROOT, check_backup


DOCX = ROOT / "Harish_Bharathwaj_230953594_FGAI_Assignment3_Report.docx"
PDF = ROOT / "Harish_Bharathwaj_230953594_FGAI_Assignment3_Report.pdf"
FONT = Path("C:/Windows/Fonts/arial.ttf")
BOLD_FONT = Path("C:/Windows/Fonts/arialbd.ttf")


def metric_rows(evidence):
    rows = []
    for name in ("basic", "mt5", "llama", "gemma"):
        if name == "mt5":
            rows.append(["mT5-small LoRA", "Failed", "-", "-", "Not saved", "-", "Not saved"])
            continue
        m = evidence["metrics"][name]
        rows.append([MODEL_LABELS[name], "Completed", f"{m['bleu']:.3f}",
                     f"{m['chrf']:.3f}",
                     f"{m['training_seconds']:.2f}" if m["training_seconds"] is not None else "N/A",
                     f"{m['inference_seconds_per_sentence']:.3f}",
                     f"{m['parameters']:,}"])
    return rows


def temperature_rows(evidence):
    return [[MODEL_LABELS[name], f"{item['temperature']:.1f}",
             f"{item['bleu_on_8']:.3f}", f"{item['chrf_on_8']:.3f}",
             str(item["empty_outputs"])]
            for name in ("basic", "llama", "gemma")
            for item in evidence["temperatures"][name]]


def text_blocks(evidence):
    m = evidence["metrics"]
    return {
        "summary": ("Three systems completed 200-item evaluation on an identical held-out test set. "
                    "Llama has the highest recorded BLEU and chrF among them. mT5-small completed two "
                    "finite-loss LoRA epochs but failed during generation when it emitted a sentinel token "
                    "despite suppression. Its translation quality and inference cost remain unmeasured."),
        "dataset": ("The code uses OPUS Books English-French parallel text, normalized and filtered by "
                    "sentence length, deduplicated, then shuffled with seed 42. It specifies 4,000 training, "
                    "400 validation, and 200 test pairs. The three completed models have the same 200 "
                    "source/reference rows. Reconstructing their test CSV reproduces the archived SHA-256 "
                    f"{evidence['test_hash']}. The original split CSVs and dataset manifest were absent from "
                    "the backup, so train/validation contents could not be rechecked."),
        "method": ("The Basic system is a two-layer, four-head encoder-decoder Transformer trained from "
                   "scratch for 10 epochs. mT5-small used rank-8 LoRA for two epochs in FP32. Llama 3.2 1B "
                   "Instruct and Gemma 3 270M IT were prompted zero-shot. Primary generation used greedy "
                   "decoding with at most 64 new tokens. mT5 applied sentinel and repetition controls; "
                   "causal models used a chat prompt and a plain-prompt retry for empty chat output."),
        "metric_note": ("BLEU and chrF below are the saved SacreBLEU corpus scores, not independently "
                        "recomputed in this workspace. The audit checked finite score ranges, 200 complete "
                        "translation records per completed model, identical test rows, timing arithmetic, "
                        "and digit diagnostics. Inference times exclude model loading."),
        "comparison": (f"Llama leads the completed systems at BLEU {m['llama']['bleu']:.3f} and chrF "
                       f"{m['llama']['chrf']:.3f}, with 1,235,814,400 parameters and 0.682 seconds per "
                       "sentence. Basic is smallest and fastest (4,698,250 parameters; 0.116 seconds per "
                       "sentence) but has BLEU 0.665. Gemma has 268,098,176 parameters and BLEU 6.103; "
                       "45 of its 200 outputs copy the English source exactly. These results reflect "
                       "different pretraining, sizes, tokenizers, adaptation, and prompts, so architecture "
                       "alone does not explain the ranking."),
        "training": ("Basic training and validation losses fell from 7.173/6.292 to 4.563/5.246 over "
                     "10 epochs; its recorded training time was 28.04 seconds. mT5's two finite-loss epochs "
                     "fell from 7.825/3.908 to 5.098/3.725. Its generation then raised "
                     "ValueError: mT5 emitted a sentinel token despite suppression. The backup has no "
                     "validated mT5 metrics or 200-item translation file; training time and parameter count "
                     "were not saved in a completed mT5 metrics artifact."),
        "errors": ("The saved outputs reveal concrete quality risks. Basic repeats 'le capitaine' in item "
                   "1. Llama describes Prudence as seated although the source says she was standing. "
                   "Gemma returns the English source unchanged in item 2 and does so for 45 of 200 items; "
                   "Llama has 5 exact source copies. A French reference for test item 3 contains an extra "
                   "sentence absent from its English source, indicating possible alignment noise. Among "
                   "four sources with digits, all four Basic outputs and two each from Llama and Gemma "
                   "changed digit sequences."),
        "ethics": ("Stance changes, missing translation, altered numbers, repetition, and possible corpus "
                   "misalignment can distort people, claims, and cultural context. Bias, gender, politeness, "
                   "and sensitive-content safety were not systematically measured. For public-facing or "
                   "high-stakes translation, retain the source, check names and numbers, flag source copies "
                   "and repetition, review diverse examples with a human translator, and offer correction. "
                   "BLEU and chrF measure overlap with references rather than adequacy or safety."),
        "temperature": ("The optional study sampled only the first eight shared test items at "
                        "temperatures 0.2, 0.7, and 1.0. These exploratory scores are not comparable to "
                        "the 200-item greedy results, and eight examples are too few to rank sampling "
                        "settings reliably. No mT5 temperature result exists."),
        "conclusion": ("Llama is best by the recorded BLEU and chrF among the three completed models. "
                       "The four-model assignment remains incomplete because mT5 has no validated "
                       "translation or quality score. The backup also lacks the original data manifest "
                       "and split CSVs; SacreBLEU could not be independently rerun in the audit workspace. "
                       "The archived mT5-only status is preserved, and the reconciled status explicitly "
                       "identifies the combined evidence from multiple runs."),
    }


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def docx_table(doc, headers, rows, widths):
    table = doc.add_table(rows=1, cols=len(headers))
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    for index, value in enumerate(headers):
        cell = table.rows[0].cells[index]
        cell.text = value
        cell.width = Inches(widths[index])
        set_cell_shading(cell, "E7EEF6")
        for run in cell.paragraphs[0].runs:
            run.bold = True
            run.font.size = Pt(7.4)
    for values in rows:
        row = table.add_row()
        for index, value in enumerate(values):
            cell = row.cells[index]
            cell.text = str(value)
            cell.width = Inches(widths[index])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_after = Pt(0)
                for run in paragraph.runs:
                    run.font.size = Pt(7.3)
        if values[0].startswith("mT5"):
            for cell in row.cells:
                set_cell_shading(cell, "F8ECEC")
    doc.add_paragraph().paragraph_format.space_after = Pt(0)
    return table


def build_docx(evidence):
    body = text_blocks(evidence)
    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Inches(8.27), Inches(11.69)
    section.top_margin = section.bottom_margin = Inches(0.65)
    section.left_margin = section.right_margin = Inches(0.65)
    styles = doc.styles
    for key in ("Normal", "Title", "Heading 1", "Heading 2"):
        style = styles[key]
        style.font.name = "Arial"
        style.font.color.rgb = RGBColor(0, 0, 0)
    styles["Normal"].font.size = Pt(9)
    styles["Normal"].paragraph_format.space_after = Pt(5)
    styles["Normal"].paragraph_format.line_spacing = 1.08
    styles["Title"].font.size = Pt(20)
    styles["Title"].font.bold = True
    styles["Heading 1"].font.size = Pt(12)
    styles["Heading 1"].font.bold = True
    styles["Heading 1"].paragraph_format.space_before = Pt(12)
    styles["Heading 1"].paragraph_format.space_after = Pt(5)
    styles["Heading 1"].paragraph_format.keep_with_next = True
    styles["Heading 2"].font.size = Pt(10)
    styles["Heading 2"].font.bold = True
    styles["Heading 2"].paragraph_format.keep_with_next = True

    doc.add_paragraph("English to French Machine Translation Comparison", "Title")
    p = doc.add_paragraph()
    p.add_run("ICT4403 Foundations of Generative AI Assignment 3").bold = True
    doc.add_paragraph("Harish Bharathwaj  |  Registration 230953594")
    doc.add_paragraph(body["summary"])

    doc.add_heading("Dataset and methods", 1)
    doc.add_paragraph(body["dataset"])
    doc.add_paragraph(body["method"])
    doc.add_heading("Measured comparison", 1)
    docx_table(doc, ["System", "Status", "BLEU", "chrF", "Train s", "Infer s/item", "Parameters"],
               metric_rows(evidence), [1.42, 0.78, 0.52, 0.52, 0.68, 0.85, 1.20])
    doc.add_paragraph(body["metric_note"])
    doc.add_paragraph(body["comparison"])
    doc.add_picture(str(ROOT / "results" / "comparison.png"), width=Inches(6.75))
    doc.add_paragraph("Figure 1. Recorded scores and inference time for completed models only.")

    doc.add_heading("Training evidence and failure", 1)
    doc.add_paragraph(body["training"])
    doc.add_picture(str(ROOT / "results" / "training_curves.png"), width=Inches(5.7))
    doc.add_paragraph("Figure 2. Basic Transformer training and validation loss; mT5 has no validated evaluation.")

    doc.add_heading("Saved translation examples", 1)
    doc.add_paragraph("Predictions below are verbatim from the shared test set.")
    for index in (0, 1, 3):
        sample = evidence["translations"]["basic"][index]
        doc.add_heading(f"Test item {index + 1}", 2)
        for label, text in [("English", sample["source"]), ("Reference French", sample["reference"]),
                            *[(MODEL_LABELS[name], evidence["translations"][name][index]["prediction"])
                              for name in ("basic", "llama", "gemma")]]:
            paragraph = doc.add_paragraph()
            paragraph.add_run(label + ": ").bold = True
            paragraph.add_run(text)

    doc.add_heading("Error analysis and responsible use", 1)
    doc.add_paragraph(body["errors"])
    doc.add_paragraph(body["ethics"])
    doc.add_heading("Sampling temperature analysis", 1)
    doc.add_paragraph(body["temperature"])
    docx_table(doc, ["Model", "Temp", "BLEU on 8", "chrF on 8", "Empty"],
               temperature_rows(evidence), [2.2, 0.6, 1.15, 1.15, 0.65])
    doc.add_heading("Conclusion and limits", 1)
    doc.add_paragraph(body["conclusion"])
    doc.add_heading("Sources and reproducibility", 1)
    doc.add_paragraph("Dataset: https://huggingface.co/datasets/Helsinki-NLP/opus_books")
    doc.add_paragraph("Models: https://huggingface.co/google/mt5-small ; https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct ; https://huggingface.co/google/gemma-3-270m-it")
    doc.add_paragraph("Repository supplied by student: https://github.com/Harish766/GenAI2.git")
    doc.add_paragraph("Test split SHA-256: " + evidence["test_hash"])
    doc.add_paragraph("Evidence: results/REPORT.md, results/validation.json, results/run_status.json, saved metrics, translations, plots, and the archived mT5 attempt status.")
    doc.save(DOCX)
    return DOCX


def build_pdf(evidence):
    body = text_blocks(evidence)
    if not (FONT.is_file() and BOLD_FONT.is_file()):
        raise FileNotFoundError("Arial TTF fonts are required for accented French PDF text")
    pdfmetrics.registerFont(TTFont("ArialReport", str(FONT)))
    pdfmetrics.registerFont(TTFont("ArialReportBold", str(BOLD_FONT)))
    pdfmetrics.registerFontFamily("ArialReport", normal="ArialReport", bold="ArialReportBold")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", fontName="ArialReportBold", fontSize=18,
                              leading=22, textColor=colors.black, spaceAfter=12))
    styles.add(ParagraphStyle(name="ReportHeading", fontName="ArialReportBold", fontSize=12,
                              leading=15, textColor=colors.black, spaceBefore=12, spaceAfter=5,
                              keepWithNext=True))
    styles.add(ParagraphStyle(name="ReportSubheading", fontName="ArialReportBold", fontSize=10,
                              leading=13, textColor=colors.black, spaceBefore=8, spaceAfter=4,
                              keepWithNext=True))
    styles.add(ParagraphStyle(name="ReportBody", fontName="ArialReport", fontSize=8.8,
                              leading=12.1, textColor=colors.black, spaceAfter=7))
    styles.add(ParagraphStyle(name="ReportSmall", fontName="ArialReport", fontSize=7.3,
                              leading=9.6, textColor=colors.black))
    styles.add(ParagraphStyle(name="ReportCaption", fontName="ArialReport", fontSize=7.7,
                              leading=10, textColor=colors.HexColor("#475569"), spaceAfter=8))
    para = lambda value, style="ReportBody": Paragraph(escape(value), styles[style])
    heading = lambda value: Paragraph(escape(value), styles["ReportHeading"])
    story = [para("English to French Machine Translation Comparison", "ReportTitle"),
             para("ICT4403 Foundations of Generative AI Assignment 3 | Harish Bharathwaj | 230953594"),
             para(body["summary"]), heading("Dataset and methods"),
             para(body["dataset"]), para(body["method"]), heading("Measured comparison")]

    def pdf_table(headers, values, widths):
        content = [[para(x, "ReportSmall") for x in headers]]
        content += [[para(str(x), "ReportSmall") for x in row] for row in values]
        table = Table(content, colWidths=[x * inch for x in widths], repeatRows=1, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e7eef6")),
            ("BACKGROUND", (0, 2), (-1, 2), colors.HexColor("#f8ecec")) if len(headers) == 7 else
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e7eef6")),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        return table

    story += [pdf_table(["System", "Status", "BLEU", "chrF", "Train s", "Infer s/item", "Parameters"],
                        metric_rows(evidence), [1.36, 0.72, 0.52, 0.52, 0.66, 0.80, 1.20]),
              Spacer(1, 9), para(body["metric_note"]), para(body["comparison"]),
              Image(str(ROOT / "results" / "comparison.png"), width=6.75*inch, height=2.05*inch),
              para("Figure 1. Recorded scores and inference time for completed models only.", "ReportCaption"),
              heading("Training evidence and failure"), para(body["training"]),
              Image(str(ROOT / "results" / "training_curves.png"), width=5.8*inch, height=3.43*inch),
              para("Figure 2. Basic Transformer loss history; mT5 has no validated evaluation.", "ReportCaption"),
              heading("Saved translation examples"),
              para("Predictions below are verbatim from the shared test set.")]
    for index in (0, 1, 3):
        sample = evidence["translations"]["basic"][index]
        block = [para(f"Test item {index + 1}", "ReportSubheading")]
        for label, value in [("English", sample["source"]), ("Reference French", sample["reference"]),
                             *[(MODEL_LABELS[name], evidence["translations"][name][index]["prediction"])
                               for name in ("basic", "llama", "gemma")]]:
            block.append(para(f"{label}: {value}"))
        story.append(KeepTogether(block))
    story += [heading("Error analysis and responsible use"), para(body["errors"]),
              para(body["ethics"]), heading("Sampling temperature analysis"),
              para(body["temperature"]),
              pdf_table(["Model", "Temp", "BLEU on 8", "chrF on 8", "Empty"],
                        temperature_rows(evidence), [2.20, 0.64, 1.12, 1.12, 0.68]),
              Spacer(1, 8), heading("Conclusion and limits"), para(body["conclusion"]),
              heading("Sources and reproducibility"),
              para("Dataset: https://huggingface.co/datasets/Helsinki-NLP/opus_books"),
              para("Models: https://huggingface.co/google/mt5-small ; https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct ; https://huggingface.co/google/gemma-3-270m-it"),
              para("Repository supplied by student: https://github.com/Harish766/GenAI2.git"),
              para("Test split SHA-256: " + evidence["test_hash"]),
              para("Evidence: results/REPORT.md, results/validation.json, results/run_status.json, saved metrics, translations, plots, and the archived mT5 attempt status.")]
    def footer(canvas, document):
        canvas.setFont("ArialReport", 8)
        canvas.setFillColor(colors.HexColor("#475569"))
        canvas.drawRightString(A4[0] - .65*inch, .42*inch, f"Page {document.page}")
    doc = SimpleDocTemplate(str(PDF), pagesize=A4,
                            leftMargin=.65*inch, rightMargin=.65*inch,
                            topMargin=.65*inch, bottomMargin=.65*inch)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return PDF


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=("docx", "pdf"), required=True)
    args = parser.parse_args()
    evidence = check_backup()
    print(build_docx(evidence) if args.format == "docx" else build_pdf(evidence))
