from __future__ import annotations

import argparse
import html
import shutil
import tempfile
import zipfile
from pathlib import Path


CAPTION_PLACEHOLDER = "results of nme and proposed method"
CAPTION_TEXT = "Table 1. Average results (%) across seven CIL methods."


ROWS = (
    ("Dataset", "Metric", "NME", "Proposed", "Gain"),
    ("CIFAR-100", "AIA", "57.191", "57.648", "+0.457"),
    ("CIFAR-100", "Final", "46.346", "46.907", "+0.561"),
    ("ImageNet-100", "AIA", "64.900", "65.260", "+0.360"),
    ("ImageNet-100", "Final", "54.726", "55.260", "+0.534"),
)


COLUMN_WIDTHS = (1300, 800, 850, 1150, 796)
TABLE_WIDTH = sum(COLUMN_WIDTHS)


def _run_properties(*, bold: bool) -> str:
    bold_xml = "<w:b/><w:bCs/>" if bold else ""
    return (
        "<w:rPr>"
        '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" '
        'w:eastAsia="Times New Roman" w:cs="Times New Roman"/>'
        f"{bold_xml}"
        '<w:sz w:val="15"/><w:szCs w:val="15"/>'
        "</w:rPr>"
    )


def _cell(text: str, width: int, *, header: bool, bold: bool) -> str:
    fill = '<w:shd w:val="clear" w:color="auto" w:fill="D9EAF7"/>' if header else ""
    return (
        "<w:tc>"
        "<w:tcPr>"
        f'<w:tcW w:w="{width}" w:type="dxa"/>'
        f"{fill}"
        '<w:vAlign w:val="center"/>'
        "</w:tcPr>"
        "<w:p>"
        "<w:pPr>"
        '<w:spacing w:before="0" w:after="0" w:line="170" w:lineRule="auto"/>'
        '<w:jc w:val="center"/>'
        "</w:pPr>"
        f"<w:r>{_run_properties(bold=bold)}<w:t>{html.escape(text)}</w:t></w:r>"
        "</w:p>"
        "</w:tc>"
    )


def _table_xml() -> str:
    borders = "".join(
        f'<w:{edge} w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
        for edge in ("top", "left", "bottom", "right", "insideH", "insideV")
    )
    grid = "".join(f'<w:gridCol w:w="{width}"/>' for width in COLUMN_WIDTHS)
    body_rows: list[str] = []
    for row_index, values in enumerate(ROWS):
        cells = []
        for column_index, (value, width) in enumerate(zip(values, COLUMN_WIDTHS)):
            cells.append(
                _cell(
                    value,
                    width,
                    header=row_index == 0,
                    bold=(row_index == 0 or column_index in {3, 4}),
                )
            )
        body_rows.append(
            '<w:tr><w:trPr><w:cantSplit/></w:trPr>'
            + "".join(cells)
            + "</w:tr>"
        )
    return (
        "<w:tbl>"
        "<w:tblPr>"
        f'<w:tblW w:w="{TABLE_WIDTH}" w:type="dxa"/>'
        '<w:tblInd w:w="0" w:type="dxa"/>'
        '<w:tblLayout w:type="fixed"/>'
        f"<w:tblBorders>{borders}</w:tblBorders>"
        "<w:tblCellMar>"
        '<w:top w:w="20" w:type="dxa"/><w:left w:w="35" w:type="dxa"/>'
        '<w:bottom w:w="20" w:type="dxa"/><w:right w:w="35" w:type="dxa"/>'
        "</w:tblCellMar>"
        '<w:tblLook w:val="04A0" w:firstRow="1" w:lastRow="0" '
        'w:firstColumn="0" w:lastColumn="0" w:noHBand="0" w:noVBand="1"/>'
        "</w:tblPr>"
        f"<w:tblGrid>{grid}</w:tblGrid>"
        + "".join(body_rows)
        + "</w:tbl>"
    )


def _replace_document_xml(document_xml: str) -> str:
    if CAPTION_PLACEHOLDER not in document_xml:
        raise RuntimeError(f"caption placeholder not found: {CAPTION_PLACEHOLDER!r}")
    if "Table 1. Average results (%) across seven CIL methods." in document_xml:
        raise RuntimeError("the results table caption is already present")

    marker = document_xml.index(CAPTION_PLACEHOLDER)
    paragraph_start = document_xml.rfind("<w:p ", 0, marker)
    if paragraph_start < 0:
        paragraph_start = document_xml.rfind("<w:p>", 0, marker)
    paragraph_end = document_xml.index("</w:p>", marker) + len("</w:p>")
    caption_paragraph = document_xml[paragraph_start:paragraph_end]
    updated_caption = caption_paragraph.replace(CAPTION_PLACEHOLDER, CAPTION_TEXT)
    if updated_caption == caption_paragraph:
        raise RuntimeError("failed to update the results table caption")

    return (
        document_xml[:paragraph_start]
        + updated_caption
        + _table_xml()
        + document_xml[paragraph_end:]
    )


def insert_results_table(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)

    with zipfile.ZipFile(destination, "r") as archive:
        original_xml = archive.read("word/document.xml").decode("utf-8")
        updated_xml = _replace_document_xml(original_xml)

    with tempfile.NamedTemporaryFile(
        dir=destination.parent, suffix=".docx", delete=False
    ) as handle:
        temporary = Path(handle.name)

    try:
        with zipfile.ZipFile(destination, "r") as source_archive, zipfile.ZipFile(
            temporary, "w"
        ) as target_archive:
            for item in source_archive.infolist():
                payload = (
                    updated_xml.encode("utf-8")
                    if item.filename == "word/document.xml"
                    else source_archive.read(item.filename)
                )
                target_archive.writestr(item, payload)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    insert_results_table(args.source.resolve(), args.destination.resolve())
    print(args.destination.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
