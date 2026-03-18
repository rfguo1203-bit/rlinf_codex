import importlib.util
import pathlib


SCRIPT_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "examples"
    / "embodiment"
    / "eval_libero10_pi05_minimal.py"
)

spec = importlib.util.spec_from_file_location("eval_libero10_pi05_minimal", SCRIPT_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_compute_num_save_videos():
    assert module.compute_num_save_videos(200, 0.2) == 40
    assert module.compute_num_save_videos(10, 0.0) == 0
    assert module.compute_num_save_videos(10, 1.0) == 10


def test_select_video_indices_even_spacing():
    indices = module.select_video_indices(10, 3)
    assert indices == [0, 4, 9]


def test_build_trial_specs_is_deterministic_and_unique():
    trial_specs = module.build_trial_specs([2, 3], total_trials=4, shuffle_seed=0)
    assert len(trial_specs) == 4
    assert len(set(trial_specs)) == 4
    assert trial_specs == [(1, 0), (0, 1), (0, 0), (1, 2)]
