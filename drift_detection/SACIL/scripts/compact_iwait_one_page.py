"""Create the compact one-page IWAIT draft from the researcher's v0.2 DOCX."""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path
import re
import zipfile


ABSTRACT = (
    "Class-incremental learning (CIL) learns new classes while retaining old "
    "knowledge. In exemplar-based CIL, old training data are replaced by a "
    "small exemplar memory. Nearest-mean-of-exemplars (NME) classifies samples "
    "using prototypes computed as exemplar-feature means. However, these means "
    "can deviate from unavailable full-data means, whose use yields higher "
    "accuracy in our oracle study. We therefore predict current full-data "
    "prototypes by transforming their introduction-time values according to "
    "inter-session feature changes measured with retained exemplars. Across "
    "seven learners under B50-Inc5, our method improves average incremental "
    "accuracy (AIA) over NME in all 14 CIFAR-100 and ImageNet-100 evaluations, "
    "by 0.457 and 0.360 percentage points on average, respectively."
)

INTRODUCTION = (
    "CIL must learn new classes without forgetting previously learned ones "
    "[1]. In exemplar-based CIL, only a small subset of old-class data is "
    "retained, and nearest-mean-of-exemplars (NME) represents each class by the "
    "mean feature of these exemplars [2]. Because this subset may not fully "
    "represent the class distribution, its prototype can deviate from the "
    "unavailable full-data prototype, particularly as the feature extractor "
    "evolves across sessions. Our oracle study confirms that full-data "
    "prototypes provide higher accuracy. We therefore predict their current "
    "positions from feature-space changes observed through retained exemplars, "
    "providing a post-hoc estimator applicable to different CIL learners."
)

PROPOSED_METHOD = (
    "When a class is introduced, we store its full-data prototype. After each "
    "session, the same retained exemplars are embedded by the previous and "
    "current models, producing paired features that estimate a "
    "ridge-regularized transformation between consecutive feature spaces. The "
    "transformation recursively moves stored old-class prototypes into the "
    "current space. At inference, each current-model feature is assigned to its "
    "nearest normalized prototype, using transformed prototypes for old "
    "classes and exemplar means for newly introduced classes. This post-hoc "
    "process leaves CIL training unchanged."
)

EXPERIMENTAL_RESULTS = (
    "We evaluate CIFAR-100/ResNet-32 and ImageNet-100/ResNet-18 under B50-Inc5 "
    "with 20 exemplars per class and seed 1. For each learner, NME and our "
    "estimator use the same checkpoint, memory, class order, and test set. AIA "
    "is the mean accuracy over all sessions, and Last is the final-session "
    "accuracy. As shown in Tables 1 and 2, AIA improves in all 14 pairs, with "
    "average gains of 0.457 and 0.360 percentage points on CIFAR-100 and "
    "ImageNet-100, respectively."
)

CONCLUSION = (
    "We introduced a post-hoc estimator that tracks unavailable full-data "
    "prototypes from feature changes of retained exemplars. Its AIA gains in "
    "all 14 evaluations support prototype estimation as a complement to CIL "
    "training; multi-seed validation remains future work."
)

SURVEY_REFERENCE = (
    "M. Masana, X. Liu, B. Twardowski, M. Menta, A. D. Bagdanov, and J. van de "
    "Weijer, “Class-Incremental Learning: Survey and Performance Evaluation on "
    "Image Classification,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 45, "
    "no. 5, pp. 5513–5533, May 2023."
)

ICARL_REFERENCE = (
    "S.-A. Rebuffi, A. Kolesnikov, G. Sperl, and C. H. Lampert, “iCaRL: "
    "Incremental Classifier and Representation Learning,” in Proc. IEEE Conf. "
    "Comput. Vis. Pattern Recognit. (CVPR), pp. 2001–2010, 2017."
)


def _paragraph_bounds(xml: str, needle: str) -> tuple[int, int, str]:
    index = xml.find(needle)
    if index < 0:
        raise ValueError(f"Could not find paragraph containing {needle!r}")
    start = max(xml.rfind("<w:p ", 0, index), xml.rfind("<w:p>", 0, index))
    end = xml.find("</w:p>", index)
    if start < 0 or end < 0:
        raise ValueError(f"Could not isolate paragraph containing {needle!r}")
    end += len("</w:p>")
    return start, end, xml[start:end]


def _replace_paragraph(
    xml: str,
    needle: str,
    runs: str,
    *,
    preserve_picture: bool = False,
) -> str:
    start, end, paragraph = _paragraph_bounds(xml, needle)
    opening_end = paragraph.find(">") + 1
    opening = paragraph[:opening_end]
    ppr_match = re.search(r"<w:pPr\b.*?</w:pPr>", paragraph, flags=re.DOTALL)
    ppr = ppr_match.group(0) if ppr_match else ""
    if preserve_picture:
        pict_start = paragraph.find("<w:pict")
        if pict_start < 0:
            raise ValueError(f"No picture found in paragraph containing {needle!r}")
        run_start = max(
            paragraph.rfind("<w:r ", 0, pict_start),
            paragraph.rfind("<w:r>", 0, pict_start),
        )
        run_end = paragraph.find("</w:r>", pict_start)
        if run_start < 0 or run_end < 0:
            raise ValueError("Could not isolate the picture run")
        runs = paragraph[run_start : run_end + len("</w:r>")] + runs
    return xml[:start] + f"{opening}{ppr}{runs}</w:p>" + xml[end:]


def _text_run(text: str) -> str:
    return f"<w:r><w:t>{escape(text)}</w:t></w:r>"


def _reference_run(text: str) -> str:
    return (
        '<w:r><w:rPr><w:sz w:val="16"/><w:szCs w:val="16"/></w:rPr>'
        f"<w:t>{escape(text)}</w:t></w:r>"
    )


def _abstract_runs(body: str) -> str:
    return (
        '<w:r><w:rPr><w:b/><w:bCs/></w:rPr><w:t>Abstract—</w:t></w:r>'
        f'<w:r><w:t xml:space="preserve"> {escape(body)}</w:t></w:r>'
    )


def transform_document_xml(xml: str) -> str:
    xml = _replace_paragraph(
        xml,
        "Class-incremental learning (CIL) sequentially",
        _abstract_runs(ABSTRACT),
        preserve_picture=True,
    )
    xml = _replace_paragraph(xml, "Class-incremental learning aims", _text_run(INTRODUCTION))
    xml = _replace_paragraph(xml, "Our method predicts", _text_run(PROPOSED_METHOD))
    xml = _replace_paragraph(xml, "We evaluate the proposed prototype", _text_run(EXPERIMENTAL_RESULTS))
    xml = _replace_paragraph(xml, "This work addresses", _text_run(CONCLUSION))
    xml = _replace_paragraph(
        xml,
        "results of nme and proposed method",
        _text_run("Table 1. CIFAR-100 results: AIA / Last (%)."),
    )
    xml = _replace_paragraph(
        xml,
        "results of nme and proposed method",
        _text_run("Table 2. ImageNet-100 results: AIA / Last (%)."),
    )
    xml = _replace_paragraph(xml, "M. Masana et al.", _reference_run(SURVEY_REFERENCE))
    xml = _replace_paragraph(xml, "Rebuffi", _reference_run(ICARL_REFERENCE))
    return xml


def rewrite_docx(source: Path, destination: Path) -> None:
    if source.resolve() == destination.resolve():
        raise ValueError("Source and destination must differ")
    with zipfile.ZipFile(source, "r") as input_zip:
        xml = input_zip.read("word/document.xml").decode("utf-8")
        xml = transform_document_xml(xml)
        with zipfile.ZipFile(destination, "w") as output_zip:
            for item in input_zip.infolist():
                data = (
                    xml.encode("utf-8")
                    if item.filename == "word/document.xml"
                    else input_zip.read(item.filename)
                )
                output_zip.writestr(item, data)
    with zipfile.ZipFile(destination, "r") as output_zip:
        bad_member = output_zip.testzip()
        if bad_member is not None:
            raise RuntimeError(f"Corrupt DOCX member: {bad_member}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    rewrite_docx(args.source, args.destination)
    print(args.destination)


if __name__ == "__main__":
    main()
