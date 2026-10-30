"""Render the cover DOCX content to a matching one-page PDF.

This reads the saved Word document's title, introduction, table, and closing
paragraph so the PDF stays synchronized with edits to the cover sheet.
"""
from pathlib import Path
from xml.sax.saxutils import escape

from docx import Document
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

source = Path("Harish_Bharathwaj_230953594_FGAI_Assignment3.docx")
target = source.with_suffix(".pdf")
doc = Document(source)
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="CoverTitle", parent=styles["Title"], fontName="Helvetica-Bold",
                          fontSize=18, leading=23, alignment=TA_CENTER, textColor=colors.black,
                          spaceAfter=15))
styles.add(ParagraphStyle(name="CoverSubtitle", parent=styles["Normal"], fontName="Helvetica-Bold",
                          fontSize=10, leading=14, alignment=TA_CENTER, spaceAfter=22))
styles.add(ParagraphStyle(name="CoverBody", parent=styles["Normal"], fontName="Helvetica",
                          fontSize=10, leading=15, spaceAfter=14))
styles.add(ParagraphStyle(name="CoverCell", parent=styles["Normal"], fontName="Helvetica",
                          fontSize=9.5, leading=13))
styles.add(ParagraphStyle(name="CoverLabel", parent=styles["CoverCell"], fontName="Helvetica-Bold"))

story = [Paragraph(escape(doc.paragraphs[0].text), styles["CoverTitle"]),
         Paragraph(escape(doc.paragraphs[1].text), styles["CoverSubtitle"]),
         Paragraph(escape(doc.paragraphs[2].text), styles["CoverBody"]),
         Spacer(1, 12)]

data = []
for row in doc.tables[0].rows:
    label, value = [cell.text for cell in row.cells]
    if value.startswith("https://"):
        shown = escape(value)
        value = f'<link href="{shown}" color="#155eaa">{shown}</link>'
    else:
        value = escape(value)
    data.append([Paragraph(escape(label), styles["CoverLabel"]),
                 Paragraph(value, styles["CoverCell"])])
table = Table(data, colWidths=[1.55*inch, 4.75*inch], hAlign="CENTER")
table.setStyle(TableStyle([
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#e8eff8")),
    ("ROWBACKGROUNDS", (1,0), (1,-1), [colors.whitesmoke, colors.white]),
    ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#ccd6e0")),
    ("LEFTPADDING", (0,0), (-1,-1), 9),
    ("RIGHTPADDING", (0,0), (-1,-1), 9),
    ("TOPPADDING", (0,0), (-1,-1), 10),
    ("BOTTOMPADDING", (0,0), (-1,-1), 10),
]))
story += [table, Spacer(1, 20), Paragraph(escape(doc.paragraphs[3].text), styles["CoverBody"])]
pdf = SimpleDocTemplate(str(target), pagesize=A4, leftMargin=0.85*inch,
                        rightMargin=0.85*inch, topMargin=0.9*inch, bottomMargin=0.8*inch)
pdf.build(story)
print(target.resolve())
