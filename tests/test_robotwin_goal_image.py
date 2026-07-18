import pytest
import torch

from cosmos_framework.data.vfm.action.goal_image import (
    cond_uses_goal_image,
    expected_goal_grid,
    expected_goal_token_count,
    goal_placeholder_string,
    merged_grid,
    preprocess_goal_image,
    validate_goal_layout,
)


@pytest.mark.parametrize(
    ("cond", "uses_goal"),
    [
        ("text_cond", False),
        ("goal_frame_cond", True),
        ("text_goal_frame_cond", True),
    ],
)
def test_goal_condition_modes(cond: str, uses_goal: bool) -> None:
    assert cond_uses_goal_image(cond) is uses_goal


@pytest.mark.parametrize(
    ("layout", "grid", "tokens"),
    [("concat", (1, 24, 20), 120), ("cam_high", (1, 16, 20), 80)],
)
def test_goal_layout_preprocessing(layout: str, grid: tuple[int, int, int], tokens: int) -> None:
    image = torch.zeros((3, 64, 64), dtype=torch.uint8)
    pixel_values, grid_thw, n_tokens = preprocess_goal_image(image, layout)

    assert expected_goal_grid(layout) == grid
    assert expected_goal_token_count(layout) == tokens
    assert tuple(grid_thw[0].tolist()) == grid
    assert n_tokens == tokens
    assert pixel_values.shape == (grid[0] * grid[1] * grid[2], 1536)
    assert merged_grid(grid_thw) == (grid[0], grid[1] // 2, grid[2] // 2)
    assert goal_placeholder_string(tokens).count("<|image_pad|>") == tokens


def test_goal_configuration_validation() -> None:
    with pytest.raises(ValueError, match="Unknown cond"):
        cond_uses_goal_image("unknown")
    with pytest.raises(ValueError, match="Unknown goal_layout"):
        validate_goal_layout("unknown")
