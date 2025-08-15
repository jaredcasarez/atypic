import cv2
import os
import numpy as np
from atypic.effects import (
    Effect,
    RollPixelsEffect,
    RandomRollPixelsEffect,
    ColorValueEffect,
    ColorChannelSplitEffect,
)
from atypic.mask import Masker, SubMasker


def test_mask_combinations():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    mask_obj = Masker(frame)
    mask_obj.create_rectangle_mask((10, 10), (40, 40))
    mask_obj.create_circle_mask((70, 70), 20)
    mask_obj.create_polygon_mask([(50, 10), (80, 10), (65, 40)])
    mask_obj.create_ellipse_mask((50, 50), (20, 10), angle=30)
    mask_obj.create_band_mask(orientation="horizontal", start=60, end=80)
    mask_obj.create_checkerboard_mask(block_size=10)
    mask_obj.create_stripe_mask(orientation="vertical", stripe_width=5, gap=5)
    print_mask(mask_obj.mask)
    mask_obj.create_full_mask()
    mask = mask_obj.mask
    assert mask.shape == (100, 100)
    assert mask.dtype == np.uint8
    assert np.any(mask == 255)


def test_effects_on_real_image():
    img_path = "input.jpg"
    assert os.path.exists(img_path)
    frame = cv2.imread(img_path)
    mask_obj = Masker(frame)
    mask_obj.create_rectangle_mask((50, 50), (150, 150))
    mask_obj.create_circle_mask((200, 200), 50)
    mask_obj.create_polygon_mask([(100, 100), (200, 100), (150, 200)])
    mask = mask_obj.mask

    # RollPixelsEffect
    effect1 = RollPixelsEffect(frame, which="row", shift_length=10, mask=mask)
    out1 = effect1.apply()
    assert out1.shape == frame.shape

    # RandomRollPixelsEffect
    effect2 = RandomRollPixelsEffect(
        frame, which="col", group_size=10, shift_range=(5, 20), mask=mask
    )
    out2 = effect2.apply()
    assert out2.shape == frame.shape

    # ColorValueEffect
    effect3 = ColorValueEffect(frame, shift_value=50, mask=mask)
    out3 = effect3.apply()
    assert out3.shape == frame.shape

    # ColorChannelSplitEffect
    effect4 = ColorChannelSplitEffect(
        frame, split_distance=15, which="row", order="bgr", mask=mask
    )
    out4 = effect4.apply()
    assert out4.shape == frame.shape


def print_mask(mask):
    """Utility function to print the mask for debugging."""
    for row in mask:
        for element in row:
            print("█" if element == 255 else "#" if element > 0 else " ", end="")
        print("")


def test_effect_mask_rectangle():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    mask_obj = Masker(frame)
    mask_obj.create_rectangle_mask((2, 2), (7, 7))
    mask = mask_obj.mask

    effect = Effect(frame, mask=mask)
    assert np.all(mask[2:7, 2:7] == 255)
    assert np.all(mask[:2, :] == 0)
    assert np.all(mask[:, :2] == 0)


def test_roll_pixels_effect_row():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    frame[5, :, :] = 1
    mask_obj = Masker(frame)
    mask_obj.create_full_mask()
    effect = RollPixelsEffect(frame, which="row", shift_length=2, mask=mask_obj.mask)
    out = effect.apply()
    assert np.array_equal(np.roll(frame[5, :, :], 2, axis=0), out[5, :, :])


def test_random_roll_pixels_row_group():
    frame = np.ones((20, 20, 3), dtype=np.uint8)
    region = (5, 5, 10, 10)
    for i in range(region[0], region[0] + region[2]):
        for j in range(region[1], region[1] + region[3]):
            frame[i, j, :] = i + j
    mask_obj = Masker(frame)
    mask_obj.create_rectangle_mask(
        (region[0], region[1]), (region[0] + region[2], region[1] + region[3])
    )
    effect = RandomRollPixelsEffect(
        frame, which="row", group_size=5, shift_range=(1, 3), mask=mask_obj.mask
    )
    out = effect.apply()
    mask = effect.mask
    assert np.any(out[mask == 255] != frame[mask == 255])
    assert np.all(out[mask == 0] == frame[mask == 0])


def test_random_roll_pixels_col_group():
    frame = np.ones((20, 20, 3), dtype=np.uint8)
    region = (5, 5, 10, 10)
    for i in range(region[0], region[0] + region[2]):
        for j in range(region[1], region[1] + region[3]):
            frame[i, j, :] = i * j
    mask_obj = Masker(frame)
    mask_obj.create_rectangle_mask(
        (region[0], region[1]), (region[0] + region[2], region[1] + region[3])
    )
    effect = RandomRollPixelsEffect(
        frame, which="col", group_size=5, shift_range=(1, 3), mask=mask_obj.mask
    )
    out = effect.apply()
    mask = effect.mask
    assert np.any(out[mask == 255] != frame[mask == 255])
    assert np.all(out[mask == 0] == frame[mask == 0])


def test_color_value_effect():
    frame = np.ones((10, 10, 3), dtype=np.uint8) * 100
    region = (2, 2, 5, 5)
    mask_obj = Masker(frame)
    mask_obj.create_rectangle_mask(
        (region[0], region[1]), (region[0] + region[2], region[1] + region[3])
    )
    effect = ColorValueEffect(frame, shift_value=50, mask=mask_obj.mask)
    out = effect.apply()
    mask = effect.mask
    assert np.all(out[mask == 255] == 150)
    assert np.all(out[mask == 0] == 100)


def test_color_channel_split_effect():
    frame = np.ones((10, 10, 3), dtype=np.uint8) * 100
    region = (2, 2, 5, 5)
    for i in range(region[0], region[0] + region[2]):
        for j in range(region[1], region[1] + region[3]):
            frame[i, j, :] = i * j
    mask_obj = Masker(frame)
    mask_obj.create_rectangle_mask(
        (region[0], region[1]), (region[0] + region[2], region[1] + region[3])
    )
    effect = ColorChannelSplitEffect(
        frame, split_distance=2, which="row", order="bgr", mask=mask_obj.mask
    )
    out = effect.apply()
    mask = effect.mask
    assert np.any(out[mask == 255] != frame[mask == 255])
    assert np.all(out[mask == 0] == frame[mask == 0])
