"""Write biblist.tex, the grouped and numbered reading list at the end of literature.pdf, from paper/references.bib.

One source for both: the paper cites references.bib through BibTeX, and this turns the same file into a list a
person can scan (grouped by strand, with a link and how far each entry was checked).

    python3 docs/overview/make_biblist.py
"""
import re
from pathlib import Path

HERE = Path(__file__).parent
BIB = HERE.parent.parent / "paper" / "references.bib"


def entries(text: str):
    group = None
    for block in re.split(r"\n(?=%\s*-{10,}|@)", text):
        m = re.match(r"%\s*-{10,}\s*(.+)", block)
        if m:
            group = m.group(1).strip()
            continue
        m = re.match(r"@(\w+)\{([^,]+),(.*)\}\s*$", block.strip(), re.S)
        if not m:
            continue
        fields = dict((k.lower(), v.strip()) for k, v in re.findall(r"(\w+)\s*=\s*\{((?:[^{}]|\{[^{}]*\})*)\}", m.group(3)))
        yield group, m.group(2), fields


def authors(a: str) -> str:
    names = [n.split(",")[0].strip() for n in a.split(" and ")]
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} \\& {names[1]}"
    return f"{names[0]} et al."


def main() -> None:
    out, n, last = [], 0, None
    for group, key, f in entries(BIB.read_text()):
        if group != last:
            if last is not None:
                out.append("\\end{enumerate}")
            out.append(f"\\subsection{{{group}}}")
            out.append(f"\\begin{{enumerate}}[label={{[\\arabic*]}},start={n + 1},leftmargin=2.2em,itemsep=2pt]")
            last = group
        n += 1
        venue = f.get("journal") or f.get("booktitle", "")
        vol = f", {f['volume']}" if "volume" in f else ""
        pages = f", {f['pages']}" if "pages" in f else ""
        link = f"\\href{{https://doi.org/{f['doi']}}}{{\\nolinkurl{{doi:{f['doi']}}}}}" if "doi" in f else (f"\\url{{{f['url']}}}" if "url" in f else "")
        status = f.get("checked", "")
        status = f" \\src{{{status}}}" if status else ""
        out.append(f"\\item {authors(f['author'])} ({f['year']}). \\textit{{{f['title']}}}{'' if f['title'].endswith('?') else '.'} {venue}{vol}{pages}. {link}{status}")
    out.append("\\end{enumerate}")
    (HERE / "biblist.tex").write_text("\n".join(out) + "\n")
    print(f"{n} entries")


if __name__ == "__main__":
    main()
