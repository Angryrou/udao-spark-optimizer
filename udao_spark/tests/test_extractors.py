import json
from pathlib import Path

import pytest

from udao_spark.data.extractors.injection_extractor import (
    get_non_decision_inputs_for_q_runtime,
    get_non_decision_inputs_for_qs_runtime,
)


@pytest.fixture
def lqp_runtime() -> str:
    with open(str(Path(__file__).parent / "assets/sample_runtime_lqp.txt")) as f:
        return f.read().strip()


def test_get_non_decision_inputs_for_lqp_runtime(lqp_runtime: str) -> None:
    d = json.loads(lqp_runtime)
    assert d["RequestType"] == "RuntimeLQP"
    non_decision_inputs = get_non_decision_inputs_for_q_runtime(d)
    print(non_decision_inputs)


@pytest.fixture
def qs_runtime() -> str:
    with open(str(Path(__file__).parent / "assets/sample_runtime_qs.txt")) as f:
        return f.read().strip()


def test_get_non_decision_inputs_for_qs_runtime(qs_runtime: str) -> None:
    d = json.loads(qs_runtime)
    assert d["RequestType"] == "RuntimeQS"
    non_decision_inputs = get_non_decision_inputs_for_qs_runtime(d, is_lqp=False)
    print(non_decision_inputs)
