"""The driver's searcher and count modes, against the real tuner on the GPU.

Slow: builds the flood driver and runs it. Deselect with `-m "not integration"`.
"""

import re
import subprocess

import pytest

from tests.test_launcher_reference import (  # noqa: F401  (built is a fixture)
    PROBLEM,
    built,
)

pytestmark = pytest.mark.integration

_COUNT = re.compile(r"^CACAO_CONFIG_COUNT (\d+)$", re.MULTILINE)


def _run(staging, duration, *extra):
    return subprocess.run(
        ["./driver", "0", "0", str(duration), "1e-6", "results", "kernels.cu",
         str(PROBLEM / "ref_cpu.c"), *extra],
        cwd=str(staging), capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=600,
    )


def _count(staging) -> int:
    proc = _run(staging, 10, "", "count")
    assert proc.returncode == 0, proc.stdout[-2000:]
    m = _COUNT.search(proc.stdout)
    assert m, proc.stdout[-2000:]
    return int(m.group(1))


def test_count_mode_prints_the_space_size_and_does_not_tune(built):
    results = built / "results.json"
    results.unlink(missing_ok=True)
    assert _count(built) > 0
    assert not results.exists()


def test_exhaustive_mode_measures_every_configuration(built):
    from utils.landscape import load_records

    n = _count(built)
    proc = _run(built, 60, "", "exhaustive")
    assert proc.returncode == 0, proc.stdout[-2000:]
    assert len(load_records(built / "results.json")) == n


def test_a_seeded_mcmc_run_measures_the_seed_first(built):
    """Finalize seeds MCMC with the source's best so the long run starts from it.

    Every configuration takes a turn as the seed, so a searcher that ignored the seed
    and started at random would pass all four with odds of 1 in 256.
    """
    import json

    from utils.landscape import Record, load_records

    proc = _run(built, 60, "", "exhaustive")
    assert proc.returncode == 0, proc.stdout[-2000:]
    data = json.loads((built / "results.json").read_text(encoding="utf-8"))
    valid = [r for r in data["Results"] if r.get("Status") == "Ok"]
    assert len(valid) > 1, proc.stdout[-2000:]
    try:
        for result in valid:
            # A seed file holding one result makes that result the best one in it.
            (built / "seed.json").write_text(json.dumps(dict(data, Results=[result])), encoding="utf-8")
            proc = _run(built, 10, "", "mcmc", "seed")
            assert proc.returncode == 0, proc.stdout[-2000:]
            first = load_records(built / "results.json")[0]
            assert first.config == Record(result).config, proc.stdout[-2000:]
    finally:
        (built / "seed.json").unlink(missing_ok=True)


def test_mcmc_mode_runs_seeded_and_unseeded(built):
    from utils.landscape import load_records

    assert _run(built, 10).returncode == 0
    for seed in ("results", ""):
        (built / "results.json").unlink(missing_ok=True) if seed == "" else None
        proc = _run(built, 10, "", "mcmc", seed)
        assert proc.returncode == 0, proc.stdout[-2000:]
        assert load_records(built / "results.json")
