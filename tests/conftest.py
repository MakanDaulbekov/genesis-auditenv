import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auditenv.env.core import AuditEnv  # noqa: E402
from auditenv.grader.service import GraderClient  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return json.loads((ROOT / "configs" / "default.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def clean_cfg(cfg):
    c = json.loads(json.dumps(cfg))
    c["clean"] = True
    return c


@pytest.fixture(scope="session")
def grader(tmp_path_factory):
    log = tmp_path_factory.mktemp("glog") / "grader_log.jsonl"
    g = GraderClient(episodes_root=str(ROOT / "episodes"), grader_root=str(ROOT / "grader_data"), log_path=str(log))
    g.log_path = log
    yield g
    g.close()


@pytest.fixture
def env(grader):
    return AuditEnv(grader, mode="train")


def act(name, **args):
    return {"action": name, "args": args}


def find_seed(cfg, pred, start=0, limit=300):
    from auditenv.generator.episode import generate_episode
    for s in range(start, start + limit):
        ep = generate_episode(s, cfg)
        if pred(ep):
            return s, ep
    raise AssertionError("no seed satisfies predicate")
