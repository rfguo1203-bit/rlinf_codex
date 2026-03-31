import importlib.util
import pathlib

import pytest


SCRIPT_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts"
    / "infer_libero10_pi05_single_task.py"
)

spec = importlib.util.spec_from_file_location(
    "infer_libero10_pi05_single_task", SCRIPT_PATH
)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_normalize_task_name_collapses_common_separators():
    assert (
        module.normalize_task_name("  Pick_Up-the  Cup ")
        == "pick up the cup"
    )


def test_resolve_task_id_prefers_explicit_task_id():
    descriptions = ["open the drawer", "pick up the cup"]
    assert module.resolve_task_id(descriptions, task_id=1, task_name=None) == 1


def test_resolve_task_id_matches_exact_normalized_name():
    descriptions = ["Open the drawer", "Pick up the cup"]
    assert (
        module.resolve_task_id(descriptions, task_name="pick_up-the cup")
        == 1
    )


def test_resolve_task_id_matches_unique_substring():
    descriptions = ["open the drawer", "pick up the cup"]
    assert module.resolve_task_id(descriptions, task_name="drawer") == 0


def test_resolve_task_id_rejects_ambiguous_substring():
    descriptions = ["open the left drawer", "open the right drawer"]
    with pytest.raises(ValueError, match="Matched multiple tasks by substring"):
        module.resolve_task_id(descriptions, task_name="drawer")


def test_resolve_task_id_rejects_missing_task_name():
    descriptions = ["open the drawer", "pick up the cup"]
    with pytest.raises(ValueError, match="Could not find a LIBERO-10 task"):
        module.resolve_task_id(descriptions, task_name="close the fridge")
