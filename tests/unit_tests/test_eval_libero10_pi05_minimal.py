import importlib.util
import pathlib


SCRIPT_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts"
    / "simple_eval_libero10_pi05.py"
)

spec = importlib.util.spec_from_file_location("simple_eval_libero10_pi05", SCRIPT_PATH)
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


def test_build_task_reset_state_ids():
    assert module.build_task_reset_state_ids([50, 100, 150], task_id=0) == list(
        range(0, 50)
    )
    assert module.build_task_reset_state_ids([50, 100, 150], task_id=2) == list(
        range(100, 150)
    )


def test_choose_reset_state_ids_keeps_full_task_by_default():
    task_reset_state_ids = [10, 11, 12]
    assert (
        module.choose_reset_state_ids(
            task_reset_state_ids=task_reset_state_ids,
            num_episodes=None,
            shuffle=False,
            seed=0,
        )
        == task_reset_state_ids
    )


def test_choose_reset_state_ids_shuffles_deterministically():
    shuffled = module.choose_reset_state_ids(
        task_reset_state_ids=[0, 1, 2, 3, 4],
        num_episodes=3,
        shuffle=True,
        seed=123,
    )
    assert shuffled == [3, 1, 4]
