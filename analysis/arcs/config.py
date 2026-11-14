"""Load config.yaml and record which parameters produced which output."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from functools import lru_cache

import yaml

from .paths import CONFIG, OUTPUTS

RUN_LOG = OUTPUTS / "run_log.json"


@lru_cache(maxsize=1)
def cfg() -> dict:
    return yaml.safe_load(CONFIG.read_text())


def config_hash() -> str:
    return hashlib.sha1(CONFIG.read_bytes()).hexdigest()[:10]


def log_run(step: str, outputs: list[str], params: dict) -> None:
    """One entry per step, replaced on rerun: date, config hash, parameters and the files written."""
    log = json.loads(RUN_LOG.read_text()) if RUN_LOG.exists() else {}
    log[step] = {"date": dt.date.today().isoformat(), "config": config_hash(), "params": params,
                 "outputs": sorted(outputs)}
    RUN_LOG.write_text(json.dumps(dict(sorted(log.items())), indent=1, sort_keys=True, default=str) + "\n")
