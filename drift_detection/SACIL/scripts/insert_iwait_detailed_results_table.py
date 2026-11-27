from __future__ import annotations

import argparse
import html
import re
import shutil
import tempfile
import zipfile
from pathlib import Path


CAPTION_PLACEHOLDER = "results of nme and proposed method"
CAPTION_TEXT = (
    "Table 1. Per-learner results (%) reported as AIA / final accuracy."
)


ROWS = (
    (
        "Learner",
        "CIFAR NME",
        "CIFAR Proposed",
        "CIFAR Gain",
        "ImageNet NME",
        "ImageNet Proposed",
        "ImageNet Gain",
    ),
    (
        "iCaRL",
        "52.422 / 44.08",
        "52.851 / 44.24",
        "+0.429 / +0.16",
        "52.732 / 43.12",
        "53.625 / 43.96",
        "+0.893 / +0.84",
    ),
    (
        "LUCIR",
        "55.742 / 43.09",
        "57.033 / 44.47",
        "+1.291 / +1.38",
        "66.275 / 54.86",
        "66.721 / 56.00",
        "+0.447 / +1.14",
    ),
    (
        "FGP-ICL",
        "56.878 / 45.62",
        "57.195 / 46.52",
        "+0.317 / +0.90",
        "65.303 / 54.56",
        "65.512 / 55.08",
        "+0.208 / +0.52",
    ),
    (
        "PODNet",
        "59.614 / 47.98",
        "59.824 / 48.30",
        "+0.209 / +0.32",
        "70.021 / 59.38",
        "70.102 / 59.46",
        "+0.082 / +0.08",
    ),
    (
        "AFC",
        "63.586 / 54.74",
        "63.609 / 54.70",
        "+0.024 / -0.04",
        "74.125 / 66.50",
        "74.447 / 66.54",
        "+0.322 / +0.04",
    ),
    (
        "CSCCT",
        "57.244 / 44.08",
        "57.629 / 44.69",
        "+0.385 / +0.61",
        "61.105 / 49.50",
        "61.339 / 49.56",
        "+0.234 / +0.06",
    ),
    (
        "CaSpeR-IL",
        "54.855 / 44.83",
        "55.396 / 45.43",
        "+0.541 / +0.60",
        "64.742 / 55.16",
        "65.077 / 56.22",
        "+0.335 / +1.06",
    ),
    (
        "Mean",
        "57.191 / 46.346",
        "57.648 / 46.907",
        "+0.457 / +0.561",
        "64.900 / 54.726",
        "65.260 / 55.260",
        "+0.360 / +0.534",
    ),
)


COLUMN_WIDTHS = (1350, 1455, 1455, 1455, 1455, 1455, 1455)
TABLE_WIDTH = sum(COLUMN_WIDTHS)


def _run_properties(*, bold: bool) -> str:
    bold_xml = "<w:b/><w:bCs/>" if bold else ""
    return (
        "<w:rPr>"
        '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" '
        'w:eastAsia="Times New Roman" w:cs="Times New Roman"/>'
        f"{bold_xml}"
        '<w:sz w:val="13"/><w:szCs w:val="13"/>'
        "</w:rPr>"
    )


def _cell(text: str, width: int, *, header: bool, bold: bool) -> str:
    fill = '<w:shd w:val="clear" w:color="auto" w:fill="D9EAF7"/>' if header else ""
    return (
        "<w:tc><w:tcPr>"
        f'<w:tcW w:w="{width}" w:type="dxa"/>{fill}'
        '<w:vAlign w:val="center"/>'
        "</w:tcPr><w:p><w:pPr>"
        '<w:spacing w:before="0" w:after="0" w:line="145" w:lineRule="auto"/>'
        '<w:jc w:val="center"/>'
        "</w:pPr>"
        f"<w:r>{_run_properties(bold=bold)}<w:t>{html.escape(text)}</w:t></w:r>"
        "</w:p></w:tc>"
    )


def _table_xml() -> str:
    borders = "".join(
        f'<w:{edge} w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
        for edge in ("top", "left", "bottom", "right", "insideH", "insideV")
    )
    grid = "".join(f'<w:gridCol w:w="{width}"/>' for width in COLUMN_WIDTHS)
    rows: list[str] = []
    for row_index, values in enumerate(ROWS):
        cells = []
        for column_index, (value, width) in enumerate(zip(values, COLUMN_WIDTHS)):
            cells.append(
                _cell(
                    value,
                    width,
                    header=row_index == 0,
                    bold=(
                        row_index == 0
                        or row_index == len(ROWS) - 1
                        or column_index in {2, 3, 5, 6}
                    ),
                )
            )
        rows.append(
            '<w:tr><w:trPr><w:cantSplit/></w:trPr>'
            + "".join(cells)
            + "</w:tr>"
        )
    return (
        "<w:tbl><w:tblPr>"
        f'<w:tblW w:w="{TABLE_WIDTH}" w:type="dxa"/>'
        '<w:tblInd w:w="0" w:type="dxa"/>'
        '<w:tblLayout w:type="fixed"/>'
        f"<w:tblBorders>{borders}</w:tblBorders>"
        "<w:tblCellMar>"
        '<w:top w:w="10" w:type="dxa"/><w:left w:w="20" w:type="dxa"/>'
        '<w:bottom w:w="10" w:type="dxa"/><w:right w:w="20" w:type="dxa"/>'
        "</w:tblCellMar>"
        '<w:tblLook w:val="04A0" w:firstRow="1" w:lastRow="0" '
        'w:firstColumn="0" w:lastColumn="0" w:noHBand="0" w:noVBand="1"/>'
        f"</w:tblPr><w:tblGrid>{grid}</w:tblGrid>"
        + "".join(rows)
        + "</w:tbl>"
    )


def _section_copy(final_section: str, *, columns: int) -> str:
    section = re.sub(
        r"^(<w:sectPr(?:\s[^>]*)?>)",
        r'\1<w:type w:val="continuous"/>',
        final_section,
        count=1,
    )
    section = re.sub(
        r"<w:cols\b[^>]*/>",
        f'<w:cols w:num="{columns}" w:space="288"/>',
        section,
        count=1,
    )
    return section


def _section_break(final_section: str, *, columns: int) -> str:
    section = _section_copy(final_section, columns=columns)
    return (
        "<w:p><w:pPr>"
        '<w:spacing w:before="0" w:after="0" w:line="1" w:lineRule="exact"/>'
        f"{section}"
        "</w:pPr></w:p>"
    )


def _replace_document_xml(document_xml: str) -> str:
    if CAPTION_PLACEHOLDER not in document_xml:
        raise RuntimeError(f"caption placeholder not found: {CAPTION_PLACEHOLDER!r}")

    sections = list(re.finditer(r"<w:sectPr\b[\s\S]*?</w:sectPr>", document_xml))
    if not sections:
        raise RuntimeError("final section properties were not found")
    final_section = sections[-1].group(0)

    marker = document_xml.index(CAPTION_PLACEHOLDER)
    paragraph_start = document_xml.rfind("<w:p ", 0, marker)
    if paragraph_start < 0:
        paragraph_start = document_xml.rfind("<w:p>", 0, marker)
    paragraph_end = document_xml.index("</w:p>", marker) + len("</w:p>")
    caption = document_xml[paragraph_start:paragraph_end].replace(
        CAPTION_PLACEHOLDER, CAPTION_TEXT
    )

    full_width_start = _section_break(final_section, columns=2)
    full_width_end = _section_break(final_section, columns=1)
    replacement = full_width_start + caption + _table_xml() + full_width_end
    return document_xml[:paragraph_start] + replacement + document_xml[paragraph_end:]


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
