"""Extract Reception / Critical response text from Wikipedia wikitext.

Run from the data folder (data/raw or build/imdb), same as parse_infobox.py:

  mkdir -p features
  python ../../scripts/imdb/extract_reception.py
  python ../../scripts/imdb/extract_reception.py wiki/articles_10k.jsonl features/wiki_reception_10k.csv
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import mwparserfromhell as M

SRC = sys.argv[1] if len(sys.argv) > 1 else "wiki/articles.jsonl"
DST = sys.argv[2] if len(sys.argv) > 2 else "features/wiki_reception.csv"

HEADING_RE = re.compile(
    r"^(?P<equals>={2,6})\s*(?P<title>.*?)\s*(?P=equals)\s*$", re.M
)
# Same family of section titles that parse_infobox.py counts as critic text.
SECTION_RE = re.compile(
    r"(?i)^(critical (?:response|reception)|reception|reviews?|critics?)$"
)
CATEGORY_YEAR_RE = re.compile(
    r"\[\[\s*Category\s*:\s*((?:19|20)\d{2})\s+films\b", re.I
)
FILM_DATE_RE = re.compile(
    r"\{\{\s*(?:Film\s*date|FilmDate)\s*\|\s*((?:19|20)\d{2})", re.I
)
YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")


def clean_wikitext(text: str) -> str:
    if not text:
        return ""
    cleaned = M.parse(text).strip_code(
        normalize=True, collapse=True, keep_template_params=False
    )
    cleaned = cleaned.replace("\xa0", " ")
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n[ \t]+", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def find_year(wikitext: str) -> tuple[int | None, str]:
    if not wikitext:
        return None, "not_found"
    m = CATEGORY_YEAR_RE.search(wikitext)
    if m:
        return int(m.group(1)), "category"
    m = FILM_DATE_RE.search(wikitext)
    if m:
        return int(m.group(1)), "film_date_template"
    infobox = re.search(
        r"\{\{\s*Infobox\s+film\b(?P<body>.*?)(?=\n\}\})",
        wikitext,
        re.I | re.S,
    )
    if infobox:
        released = re.search(
            r"\|\s*released\s*=\s*(.*?)(?=\n\|)",
            infobox.group("body"),
            re.I | re.S,
        )
        if released:
            years = YEAR_RE.findall(released.group(1))
            if years:
                return int(years[0]), "infobox_released"
    return None, "not_found"


def extract_reception(wikitext: str) -> tuple[str, str, str, list[dict], str]:
    """Return section_title, wikitext, plain text, subsections, status."""
    if not wikitext:
        return "", "", "", [], "missing_wikitext"

    headings = list(HEADING_RE.finditer(wikitext))
    reception_index = None
    section_title = ""

    for i, heading in enumerate(headings):
        title = heading.group("title").strip()
        if len(heading.group("equals")) == 2 and SECTION_RE.match(title):
            reception_index = i
            section_title = title
            break

    if reception_index is None:
        return "", "", "", [], "reception_not_found"

    start_heading = headings[reception_index]
    start = start_heading.start()
    end = len(wikitext)
    for heading in headings[reception_index + 1 :]:
        if len(heading.group("equals")) <= 2:
            end = heading.start()
            break

    section_wikitext = wikitext[start:end].strip()
    section_text = clean_wikitext(section_wikitext)

    subsections: list[dict] = []
    section_headings = [
        h
        for h in headings[reception_index + 1 :]
        if h.start() < end and len(h.group("equals")) >= 3
    ]
    for j, heading in enumerate(section_headings):
        sub_start = heading.end()
        sub_end = end
        for nxt in section_headings[j + 1 :]:
            if len(nxt.group("equals")) <= len(heading.group("equals")):
                sub_end = nxt.start()
                break
        subsections.append(
            {
                "level": len(heading.group("equals")),
                "title": heading.group("title").strip(),
                "text": clean_wikitext(wikitext[sub_start:sub_end].strip()),
            }
        )

    return section_title, section_wikitext, section_text, subsections, "ok"


def tconsts(row: dict) -> list[str]:
    raw = row.get("tconst", [])
    if isinstance(raw, list):
        return [str(x) for x in raw if x]
    if isinstance(raw, str) and raw.strip():
        return [p for p in re.split(r"[;,\s]+", raw.strip()) if p]
    return [""]


Path(DST).parent.mkdir(parents=True, exist_ok=True)
fields = [
    "tconst",
    "title",
    "year",
    "year_source",
    "section_title",
    "wikipedia_url",
    "pageid",
    "revid",
    "timestamp",
    "reception_found",
    "reception_chars",
    "reception_text",
    "reception_wikitext",
    "subsections_json",
    "extraction_status",
]

n = found = 0
with open(DST, "w", encoding="utf-8", newline="") as out:
    w = csv.DictWriter(out, fieldnames=fields)
    w.writeheader()
    for line in open(SRC, encoding="utf-8"):
        row = json.loads(line)
        title = row.get("title") or row.get("title_requested") or ""
        wikitext = row.get("wikitext") or ""
        year, year_source = find_year(wikitext)
        section_title, section_wikitext, section_text, subsections, status = (
            extract_reception(wikitext)
        )
        pageid = row.get("pageid", "")
        url = f"https://en.wikipedia.org/?curid={pageid}" if pageid else ""
        base = {
            "title": title,
            "year": year if year is not None else "",
            "year_source": year_source,
            "section_title": section_title,
            "wikipedia_url": url,
            "pageid": pageid,
            "revid": row.get("revid", ""),
            "timestamp": row.get("timestamp", ""),
            "reception_found": int(status == "ok"),
            "reception_chars": len(section_text),
            "reception_text": section_text,
            "reception_wikitext": section_wikitext,
            "subsections_json": json.dumps(subsections, ensure_ascii=False),
            "extraction_status": status,
        }
        for tc in tconsts(row):
            w.writerow({**base, "tconst": tc})
            n += 1
            found += status == "ok"

print(f"wrote {n} rows ({found} with reception) -> {DST}")
