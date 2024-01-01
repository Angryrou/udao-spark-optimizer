from pathlib import Path

import pytest


@pytest.fixture
def ckp_path_qs_lqp_compile() -> Path:
    base_dir = Path(__file__).parent
    return (
        base_dir.parent.parent
        / "playground"
        / "cache_and_ckp/tpch_22x10/qs_lqp_compile"
    )
