"""Download open-access PDFs of the papers in paper/references.bib.

For each entry with a DOI, asks OpenAlex (free, no key) for an open-access copy and saves it to
data/raw/papers/<bibkey>.pdf, which git ignores. Papers it can't fetch are listed with any open landing page
(PubMed Central and university repositories block scripts, so save those by hand) and a DTU Findit link for
the DTU library login.

    python3 scripts/get_papers.py                 # only entries with checked = {metadata}
    python3 scripts/get_papers.py --all           # every entry with a DOI
    python3 scripts/get_papers.py viana2014network wasserman2015cross

Don't commit the PDFs: they are copyrighted.
"""
import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
BIB = ROOT / "paper" / "references.bib"
UA = "Mozilla/5.0 (research; 02805-course-project)"


def field(body, name):
    m = re.search(name + r"\s*=\s*\{((?:[^{}]|\{[^{}]*\})*)\}", body)
    return m.group(1).strip() if m else None


def bib_entries():
    for block in re.split(r"\n(?=@)", BIB.read_text()):
        m = re.match(r"@\w+\{([^,]+),(.*)", block, re.S)
        if m:
            body = m.group(2)
            yield m.group(1), field(body, "doi"), field(body, "title") or "", field(body, "checked") or ""


def get(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(), r.headers.get("Content-Type", "")


def pdf_urls(doi):
    """Candidate PDF links, plus open landing pages to print if none of the PDFs download."""
    data, _ = get("https://api.openalex.org/works/doi:" + urllib.parse.quote(doi))
    w = json.loads(data)
    pdfs, pages = [], []
    for loc in [w.get("best_oa_location")] + (w.get("locations") or []):
        if not loc or not loc.get("is_oa"):
            continue
        if loc.get("pdf_url"):
            pdfs.append(loc["pdf_url"])
        pages.append(loc.get("landing_page_url") or "")
    dedup = lambda xs: [x for i, x in enumerate(xs) if x and x not in xs[:i]]
    return dedup(pdfs), dedup(pages)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("keys", nargs="*", help="bib keys to fetch (default: checked = {metadata})")
    ap.add_argument("--all", action="store_true", help="every entry with a DOI")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "raw" / "papers")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    todo = [e for e in bib_entries() if e[1]]
    if args.keys:
        todo = [e for e in todo if e[0] in args.keys]
    elif not args.all:
        todo = [e for e in todo if e[3].startswith("metadata")]

    missing = []
    for key, doi, title, _ in todo:
        dest = args.out / f"{key}.pdf"
        if dest.exists():
            print(f"have   {key}")
            continue
        try:
            urls, pages = pdf_urls(doi)
        except Exception as e:  # OpenAlex down or DOI unknown
            urls, pages = [], []
            print(f"lookup failed for {key}: {e}", file=sys.stderr)
        ok = False
        for url in urls:
            try:
                data, ctype = get(url, timeout=60)
            except Exception:
                continue
            if data[:4] == b"%PDF":
                part = dest.with_suffix(".pdf.part")
                part.write_bytes(data)
                part.rename(dest)
                print(f"saved  {key}  ({len(data) // 1024} KB from {urllib.parse.urlparse(url).netloc})")
                ok = True
                break
        if not ok:
            missing.append((key, doi, title, pages))

    if missing:
        print("\nNo open PDF found. Get these through DTU Findit (log in with your DTU account):")
        for key, doi, title, pages in missing:
            print(f"  {key}: {title}")
            for page in pages:
                print(f"    open copy (save the PDF by hand): {page}")
            print(f"    https://findit.dtu.dk/en/catalog?q={urllib.parse.quote(doi)}")
            print(f"    https://doi.org/{doi}")
    print(f"\nPDFs are in {args.out}")


if __name__ == "__main__":
    main()
