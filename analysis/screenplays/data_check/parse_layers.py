"""Split a MovieSum screenplay into layers, using only the tags it already has.

Rules (DECISIONS.md, D16, decided by the group):
    <dialogue>           -> layer "dialogue"
    <scene_description>  -> layer "description"
    <parenthetical>      -> layer "parenthetical"  (kept apart, excluded from both main layers)
    <character>          -> layer "other"
    <stage_direction>    -> layer "other"
No cleaning rules: every element is kept in the layer its tag says.
Empty elements (<dialogue />, or an empty body) are kept too, with text "",
so the statistics can count them.
"""
import json
import re
from pathlib import Path

LAYER_OF_TAG = {
    "dialogue": "dialogue",
    "scene_description": "description",
    "parenthetical": "parenthetical",
    "character": "other",
    "stage_direction": "other",
}
STRUCTURE_LINES = {"<script>", "</script>", "<scene>", "</scene>"}  # no text, no layer

FULL = re.compile(r"<(\w+)>(.*)</\1>")   # <tag>text</tag> on one line
EMPTY = re.compile(r"<(\w+) />")          # <tag />


def parse_script(script):
    """Return a list of elements: {"tag", "layer", "text", "scene"} in reading order.

    "<scene />" is kept as an element with layer "structure". Lines that match neither pattern get layer "unparsed" so nothing is lost silently.
    """
    elements = []
    scene = -1
    for line in script.split("\n"):
        line = line.strip()
        if line == "<scene>":
            scene += 1
        if line in STRUCTURE_LINES:
            continue
        if line == "<scene />":  # an empty scene marker: structure, but the scene has no content
            scene += 1
            elements.append({"tag": "scene", "layer": "structure", "text": "", "scene": scene})
            continue
        m = FULL.fullmatch(line)
        if m:
            tag, text = m.group(1), m.group(2).strip()
        else:
            m = EMPTY.fullmatch(line)
            if m:
                tag, text = m.group(1), ""
            else:
                elements.append({"tag": "?", "layer": "unparsed", "text": line, "scene": scene})
                continue
        elements.append({"tag": tag, "layer": LAYER_OF_TAG.get(tag, "unparsed"),
                         "text": text, "scene": scene})
    return elements


def read_records(data_dir):
    """Yield every record of train/val/test (the '* 2.jsonl' iCloud copies are ignored)."""
    for split in ["train", "val", "test"]:
        with open(Path(data_dir) / "scripts" / "moviesum" / f"{split}.jsonl") as f:
            for line in f:
                r = json.loads(line)
                r["split"] = split
                yield r


def one_per_id(records):
    """The group's rule (D8) for duplicated imdb ids: keep the record with the longest script
    (most characters). On a tie, the first one read is kept."""
    best = {}
    for r in records:
        old = best.get(r["imdb_id"])
        if old is None or len(r["script"]) > len(old["script"]):
            best[r["imdb_id"]] = r
    return list(best.values())
