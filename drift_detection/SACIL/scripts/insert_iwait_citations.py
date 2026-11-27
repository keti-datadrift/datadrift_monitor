"""Insert the two final citations into the current IWAIT one-page draft.

The source DOCX is preserved.  Only ``word/document.xml`` is changed in the
versioned output file, so the existing layout, tables, and embedded media stay
intact.
"""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path
import re
import zipfile


SURVEY_REFERENCE = (
    "M. Masana et al., “Class-Incremental Learning: Survey and Performance "
    "Evaluation on Image Classification,” IEEE Trans. Pattern Anal. Mach. "
    "Intell., vol. 45, no. 5, pp. 5513–5533, 2023."
)

ICARL_REFERENCE = (
    "S.-A. Rebuffi et al., “iCaRL: Incremental Classifier and Representation "
    "Learning,” Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), "
    "pp. 2001–2010, 2017."
)


def replace_paragraph_text(xml: str, needle: str, replacement: str) -> str:
    """Replace the text of the single Word paragraph containing ``needle``."""

    index = xml.find(needle)
    if index < 0:
        raise ValueError(f"Could not find reference paragraph containing {needle!r}")

    starts = (xml.rfind("<w:p ", 0, index), xml.rfind("<w:p>", 0, index))
    start = max(starts)
    end = xml.find("</w:p>", index)
    if start < 0 or end < 0:
        raise ValueError(f"Could not isolate paragraph containing {needle!r}")
    end += len("</w:p>")

    paragraph = xml[start:end]
    opening_end = paragraph.find(">") + 1
    opening = paragraph[:opening_end]
    ppr_match = re.search(r"<w:pPr\b.*?</w:pPr>", paragraph, flags=re.DOTALL)
    ppr = ppr_match.group(0) if ppr_match else ""

    # The template's reference list is 8 pt.  Explicitly retain that compact
    # size even if the two template reference paragraphs use different styles.
    run = (
        '<w:r><w:rPr><w:sz w:val="16"/><w:szCs w:val="16"/>'
        f"</w:rPr><w:t>{escape(replacement)}</w:t></w:r>"
    )
    new_paragraph = f"{opening}{ppr}{run}</w:p>"
    return xml[:start] + new_paragraph + xml[end:]


def transform_document_xml(xml: str) -> str:
    replacements = (
        (
            "previously learned ones.",
            "previously learned ones [1].",
        ),
        (
            "many exemplar-based methods employ NME, where",
            "many exemplar-based methods employ NME [2], where",
        ),
    )
    for old, new in replacements:
        count = xml.count(old)
        if count != 1:
            raise ValueError(f"Expected one occurrence of {old!r}, found {count}")
        xml = xml.replace(old, new, 1)

    xml = replace_paragraph_text(xml, "Panyaarvudh", SURVEY_REFERENCE)
    xml = replace_paragraph_text(xml, "Laerhoven", ICARL_REFERENCE)
    return xml


def rewrite_docx(source: Path, destination: Path) -> None:
    if source.resolve() == destination.resolve():
        raise ValueError("Source and destination must differ so the draft is preserved")

    with zipfile.ZipFile(source, "r") as input_zip:
        document_xml = input_zip.read("word/document.xml").decode("utf-8")
        document_xml = transform_document_xml(document_xml)

        with zipfile.ZipFile(destination, "w") as output_zip:
            for item in input_zip.infolist():
                data = (
                    document_xml.encode("utf-8")
                    if item.filename == "word/document.xml"
                    else input_zip.read(item.filename)
                )
                output_zip.writestr(item, data)

    with zipfile.ZipFile(destination, "r") as output_zip:
        bad_member = output_zip.testzip()
        if bad_member is not None:
            raise RuntimeError(f"Corrupt DOCX member after rewrite: {bad_member}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    rewrite_docx(args.source, args.destination)
    print(args.destination)


if __name__ == "__main__":
    main()
