"""Open every explorer page on this machine's data, headless, and list what didn't draw.

    .venv/bin/python -m explorer.check_pages            # every page, every notebook step, every view
    .venv/bin/python -m explorer.check_pages --quick    # each page once, on its default view

It uses Streamlit's AppTest, so no browser or server is needed. For each page and view it records crashes
(exceptions), st.error and st.warning messages, and every input file a page couldn't find (explorer.common.missing).
It also reads every file the Data inspector lists.
The report goes to the terminal and to logs/check_pages.log, and the pages' own log lines to logs/explorer.log.
Exit code 1 means at least one crash, error or unreadable file. Missing inputs are listed but don't fail it.

Notes about files in private/ (script text kept on the machine that ran a step) are listed apart, because they
are expected on every other machine.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

from explorer.log import LOG_DIR, setup  # noqa: E402

TIMEOUT = 600
PAGES = [("pages", "home"), ("pipeline_page", "pipeline"), ("pages", "text"),
         ("embedding_page", "embeddings"), ("map_pages", "maps"), ("map_pages", "checks"), ("pages", "network"), ("genre_page", "genre"),
         ("graph_pages", "whole_graph"), ("graph_pages", "movie_network"), ("graph_pages", "node"), ("graph_pages", "screenplays"),
         ("pages", "profit"), ("attribution_page", "attribution"), ("pages", "calendar"), ("pages", "tags"),
         ("pages", "tag_followups"), ("pages", "inspector"), ("pages", "checks"), ("pages", "tmdb")]
KINDS = ("CRASH", "ERROR", "MISSING", "WARNING", "PRIVATE")


class Records(logging.Handler):
    """Collects what the explorer logs while one page runs (missing inputs, private files, shown warnings)."""
    def __init__(self):
        super().__init__(logging.INFO)
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


def run_page(module: str, fn: str, state: dict | None = None, action=None) -> AppTest:
    """Run one page the way app.py does: logging set up and the page wrapped by explorer.log.logged."""
    at = AppTest.from_string(f"from explorer.log import logged, setup\nsetup()\nfrom explorer import {module}\n"
                             f"logged({module}.{fn}, {fn!r})()\n", default_timeout=TIMEOUT)
    for k, v in (state or {}).items():
        at.session_state[k] = v
    at.run()
    if action:
        action(at)
        at.run()
    return at


def findings(at: AppTest, records: list[logging.LogRecord]) -> list[tuple[str, str]]:
    out = [("CRASH", e.value) for e in at.exception]
    for r in records:
        msg = r.getMessage()
        if msg.startswith("page crashed"):
            continue                              # already listed from at.exception
        if msg.startswith("missing input: "):
            out.append(("MISSING", msg.removeprefix("missing input: ")))
        elif msg.startswith("private input"):
            out.append(("PRIVATE", msg))
        elif msg.startswith("shown: ") and r.levelno >= logging.ERROR:
            out.append(("ERROR", msg.removeprefix("shown: ")))
        elif msg.startswith("shown: "):
            out.append(("WARNING", msg.removeprefix("shown: ")))
        elif r.levelno >= logging.WARNING and r.name == "py.warnings":
            out.append(("WARNING", msg))
    return out


def views(module: str, fn: str, at: AppTest) -> list[tuple[str, dict, object]]:
    """Other views worth opening on a page: (label, session state to set first, widget action after one run)."""
    out = []
    if fn == "text":
        key = "step_scripts_text"
        radio = [r for r in at.radio if r.key == key]
        if radio:
            out += [(f"step {o}", {key: o}, None) for o in radio[0].options[1:]]
    if fn == "tags":
        ctl = [w for w in at.get("button_group") if getattr(w, "key", None) == "tag_view"]
        if ctl:
            out += [(f"view {o}", {"tag_view": o}, None) for o in ctl[0].options[1:]]
    if fn == "tag_followups":
        from explorer.pages import FOLLOW_VIEWS
        out += [(f"view {o}", {"fu_view": o}, None) for o in FOLLOW_VIEWS[1:]]
    if fn == "network":
        box = [b for b in at.selectbox if b.label == "Result file"]
        if box:
            out += [(f"result {o}", {}, lambda a, o=o: [b for b in a.selectbox if b.label == "Result file"][0]
                     .set_value(o)) for o in box[0].options[1:]]
    return out


def read_all_tables(say) -> int:
    """Read every file the Data inspector lists, as it would. Returns the number that fail."""
    from explorer.common import read_table
    from explorer.pages import data_files
    bad = 0
    for group, files in data_files().items():
        for f in files:
            try:
                nrows = 200_000 if f.stat().st_size > 150e6 else None
                df = read_table.__wrapped__(str(f), 0.0, nrows) if hasattr(read_table, "__wrapped__") \
                    else read_table(str(f), 0.0, nrows)
                if df.empty:
                    say(f"    EMPTY    {group}/{f.name}")
            except Exception as e:  # noqa: BLE001
                bad += 1
                say(f"    UNREADABLE {group}/{f.name}: {type(e).__name__}: {e}")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true", help="each page once, on its default view")
    ap.add_argument("--page", help="only pages whose function name contains this")
    args = ap.parse_args(argv)

    LOG_DIR.mkdir(exist_ok=True)
    report = LOG_DIR / "check_pages.log"
    lines: list[str] = []
    counts = {k: 0 for k in KINDS}
    setup()
    for name in ("streamlit", "streamlit.runtime.caching.cache_data_api",
                 "streamlit.runtime.scriptrunner_utils.script_run_context"):
        logging.getLogger(name).setLevel(logging.ERROR)   # bare-mode chatter, not the app's problems
    rec = Records()
    logging.getLogger("explorer").addHandler(rec)
    logging.getLogger("py.warnings").addHandler(rec)

    def say(s: str = ""):
        print(s, flush=True)
        lines.append(s)

    say(f"Explorer page check, {time.strftime('%Y-%m-%d %H:%M')}, data folder from explorer.common.DATA")
    for module, fn in PAGES:
        if args.page and args.page not in fn:
            continue
        todo = [("default view", {}, None)]
        done = 0
        while todo:
            label, state, action = todo.pop(0)
            t0 = time.time()
            rec.records.clear()
            try:
                at = run_page(module, fn, state, action)
                found = findings(at, rec.records)
            except Exception as e:  # noqa: BLE001 - a page that hangs or the harness failing is a finding too
                at, found = None, [("CRASH", f"{type(e).__name__}: {e}")]
            if done == 0 and at is not None and not args.quick:
                todo += views(module, fn, at)
            done += 1
            status = "ok" if not found else ", ".join(f"{sum(k == x for x, _ in found)} {k.lower()}"
                                                       for k in counts if any(k == x for x, _ in found))
            say(f"{fn:15s} {label[:60]:60s} {time.time() - t0:5.1f}s  {status}")
            for kind, text in found:
                counts[kind] += 1
                say(f"    {kind:8s} {' '.join(str(text).split())[:400]}")

    if not args.page:
        say()
        say("Every file the Data inspector lists:")
        bad = read_all_tables(say)
        counts["ERROR"] += bad
        say(f"    {'all readable' if not bad else f'{bad} unreadable'}")
    say()
    say("Summary: " + ", ".join(f"{v} {k.lower()}" for k, v in counts.items()))
    say("MISSING: an input file isn't on this machine; the message says how to make it.")
    say("PRIVATE: needs files kept in private/ (script text) on the machine that ran the step; expected elsewhere.")
    say("WARNING: a st.warning the page showed. Some are deliberate caveats; read them.")
    report.write_text("\n".join(lines) + "\n")
    print(f"Report written to {report.relative_to(ROOT)}")
    return 1 if counts["CRASH"] or counts["ERROR"] else 0


if __name__ == "__main__":
    sys.exit(main())
