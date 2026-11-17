"""Render the Markdown manuscript as a conference-style PDF."""

import argparse
import os
import re
from html import escape
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, KeepTogether, NextPageTemplate,
    PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("source", type=Path, help="Manuscript Markdown file.")
parser.add_argument("output", type=Path, help="Destination PDF.")
parser.add_argument("--font-dir", type=Path, default=Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts")
args = parser.parse_args()
SOURCE, OUTPUT = args.source, args.output
WIDTH, HEIGHT, LEFT, MEASURE = 612, 792, 108, 396
for family, file in (("Research", "times.ttf"), ("ResearchBold", "timesbd.ttf"),
                     ("ResearchItalic", "timesi.ttf"), ("ResearchBoldItalic", "timesbi.ttf")):
    pdfmetrics.registerFont(TTFont(family, str(args.font_dir / file)))
pdfmetrics.registerFontFamily("Research", normal="Research", bold="ResearchBold",
                              italic="ResearchItalic", boldItalic="ResearchBoldItalic")

body = ParagraphStyle("Body", fontName="Research", fontSize=10, leading=12,
                      alignment=TA_JUSTIFY, spaceAfter=7, allowWidows=0, allowOrphans=0)
abstract = ParagraphStyle("Abstract", parent=body, spaceAfter=0)
major = ParagraphStyle("Section", parent=body, fontName="ResearchBold", fontSize=12,
                       leading=15, alignment=TA_LEFT, keepWithNext=True,
                       spaceBefore=11, spaceAfter=6)
minor = ParagraphStyle("Subsection", parent=major, fontSize=10, leading=12, spaceBefore=8)
caption = ParagraphStyle("Caption", parent=body, fontSize=9.4, leading=11.2)
table_text = ParagraphStyle("Table", parent=body, alignment=TA_CENTER, fontSize=9.2,
                            leading=11, spaceAfter=0)
table_label = ParagraphStyle("TableLabel", parent=table_text, alignment=TA_LEFT)
reference = ParagraphStyle("Reference", parent=body, leftIndent=17, firstLineIndent=-17)

def inline(text):
    result = escape(text.replace("–", "-").replace("—", "-").replace("‑", "-"))
    result = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<link href="\2">\1</link>', result)
    result = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", result)
    result = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", result)
    return re.sub(r"`([^`]+)`", r'<font name="Courier" size="8.5">\1</font>', result)


blocks = re.split(r"\n\s*\n", SOURCE.read_text(encoding="utf-8-sig").strip())
title, author = blocks[0].removeprefix("# "), blocks[1]
assert blocks[2] == "## Abstract"
abstract_text = blocks[3]
assert len(abstract_text.split()) <= 200


def front(canvas, doc):
    canvas.bookmarkPage("cover")
    canvas.addOutlineEntry("Title and Abstract", "cover")
    canvas.setFillColor(colors.black)
    canvas.rect(LEFT, HEIGHT - 132.2, MEASURE, 4, fill=1, stroke=0)
    canvas.setLineWidth(1)
    canvas.line(LEFT, HEIGHT - 178.1, LEFT + MEASURE, HEIGHT - 178.1)
    canvas.setFont("ResearchBold", 17.2)
    canvas.drawCentredString(WIDTH / 2, HEIGHT - 163.8, title)
    canvas.setFont("ResearchBold", 10)
    canvas.drawCentredString(WIDTH / 2, HEIGHT - 242.8, author)
    canvas.setFont("ResearchBold", 12)
    canvas.drawCentredString(WIDTH / 2, HEIGHT - 298.5, "Abstract")
    paragraph = Paragraph(inline(abstract_text), abstract)
    _, h = paragraph.wrap(324, 270)
    paragraph.drawOn(canvas, 144, HEIGHT - 316 - h)


def footer(canvas, doc):
    canvas.setFont("Research", 10)
    canvas.drawCentredString(WIDTH / 2, 42, str(doc.page))


class Manuscript(BaseDocTemplate):
    def afterFlowable(self, flowable):
        label = getattr(flowable, "outline_label", None)
        if label:
            self.canv.bookmarkPage(label)
            self.canv.addOutlineEntry(label, label)


def add_table(block):
    rows = [[v.strip() for v in line.strip("|").split("|")] for line in block.splitlines()]
    headers, values = rows[0], rows[2:]
    if not values or any(len(row) != len(headers) for row in rows):
        raise ValueError(f"Malformed Markdown table with header {headers}.")
    cells = [[Paragraph(f"<b>{inline(cell)}</b>", table_label if i == 0 else table_text)
              for i, cell in enumerate(headers)]]
    for row in values:
        cells.append([Paragraph(inline(cell), table_label if i == 0 else table_text)
                      for i, cell in enumerate(row)])
    minimum, preferred = [], []
    for label, *entries in zip(headers, *values, strict=True):
        header_width = pdfmetrics.stringWidth(label, "ResearchBold", table_text.fontSize)
        entry_width = max(pdfmetrics.stringWidth(entry, "Research", table_text.fontSize) for entry in entries)
        token_widths = [pdfmetrics.stringWidth(token, "ResearchBold", table_text.fontSize) for token in label.split()]
        token_widths += [pdfmetrics.stringWidth(token, "Research", table_text.fontSize)
                         for value in entries for token in value.split()]
        floor = max(29, 13 + max(token_widths))
        minimum.append(floor)
        preferred.append(max(floor, min(150, 12 + max(header_width, entry_width))))
    remaining = MEASURE - sum(minimum)
    if remaining < 0:
        raise ValueError(f"Table cannot fit the manuscript width at its current font size: {headers}.")
    gaps = [want - need for want, need in zip(preferred, minimum, strict=True)]
    if not any(gaps):
        gaps = [1] * len(headers)
    widths = [need + remaining * gap / sum(gaps) for need, gap in zip(minimum, gaps, strict=True)]
    return Table(cells, colWidths=widths,
                 repeatRows=1, style=TableStyle([
                     ("VALIGN", (0, 0), (-1, -1), "TOP"),
                     ("TOPPADDING", (0, 0), (-1, -1), 3.5),
                     ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
                     ("LINEABOVE", (0, 0), (-1, 0), .7, colors.black),
                     ("LINEBELOW", (0, 0), (-1, 0), .4, colors.black),
                     ("LINEBELOW", (0, -1), (-1, -1), .7, colors.black),
                 ]))


OUTPUT.parent.mkdir(parents=True, exist_ok=True)
doc = Manuscript(str(OUTPUT), pagesize=(WIDTH, HEIGHT), title=title, author=author,
                 subject="Fine-tuning on complementary parcel-routing rulebooks")
for name, draw in (("Front", front), ("Body", footer)):
    frame = Frame(LEFT, 69, MEASURE, 651, leftPadding=0, rightPadding=0,
                  topPadding=0, bottomPadding=0)
    doc.addPageTemplates(PageTemplate(id=name, frames=frame, onPage=draw))

story = [Spacer(1, 1), NextPageTemplate("Body"), PageBreak()]
current = ""
i = 4
while i < len(blocks):
    block = blocks[i]
    if block.startswith("## "):
        current = block[3:]
        if current == "Figures":
            story.append(PageBreak())
        heading = Paragraph(inline(current), major)
        heading.outline_label = current
        story.append(heading)
    elif block.startswith("### "):
        story.append(Paragraph(inline(block[4:]), minor))
    elif block.startswith("!["):
        match = re.fullmatch(r"!\[[^\]]+\]\(([^)]+)\)", block)
        file = SOURCE.parent / match[1]
        with PILImage.open(file) as image:
            w, h = image.size
        display_width = 354 if file.name == "matched-design.png" else MEASURE
        plot = Image(str(file), width=display_width, height=display_width * h / w, hAlign="CENTER")
        i += 1
        label = Paragraph(inline(blocks[i]), caption)
        label.outline_label = blocks[i].split("**")[1].rstrip(".")
        story.append(KeepTogether([plot, Spacer(1, 4), label, Spacer(1, 12)]))
    elif block.startswith("**Table "):
        i += 1
        story.append(KeepTogether([Paragraph(inline(block), caption), add_table(blocks[i]),
                                   Spacer(1, 9)]))
    else:
        story.append(Paragraph(inline(block), reference if current == "References" else body))
    i += 1

doc.build(story)
print(OUTPUT.resolve())
