"""Tests for cutting a photographed sheet into labelled images."""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from split_sheet import (clip_to_paper, paper_mask, ruled_lines, split_groups_by_count,
                         split_rows, valley_depth)


def blob(x, w=40):
    return {"x": x, "y": 0, "w": w, "h": 50, "area": w * 50, "cy": 25.0}


def labels_of(groups):
    return [[c["x"] for c in g] for g in groups]


def test_cuts_at_the_widest_gaps_not_a_fixed_threshold():
    # "17" written with a 100 px gap inside it, then "49" only 150 px away:
    # no single threshold separates these, but the count does.
    row = [blob(0), blob(140), blob(330), blob(380)]
    assert labels_of(split_groups_by_count(row, 2)) == [[0, 140], [330, 380]]


def test_forced_cut_overrides_a_narrow_gap():
    # "96 15" written with a wide space, then "9" close behind: without help
    # the widest gap is inside 9615. Forcing a cut at x=300 fixes it.
    row = [blob(0), blob(45), blob(200), blob(245), blob(300)]
    groups = split_groups_by_count(row, 2, forced=[300])
    assert labels_of(groups) == [[0, 45, 200, 245], [300]]


def test_fewer_blobs_than_numbers_cuts_every_gap():
    row = [blob(0), blob(100)]
    assert len(split_groups_by_count(row, 3)) == 2


def test_a_zero_does_not_look_like_two_digits():
    zero = np.zeros((100, 100), np.uint8)
    cv2.circle(zero, (50, 50), 40, 255, 3)
    pair = np.zeros((100, 100), np.uint8)
    cv2.circle(pair, (25, 50), 22, 255, 3)
    cv2.circle(pair, (72, 50), 22, 255, 3)          # touching at one point
    box = (0, 0, 100, 100)
    assert valley_depth(pair, box) < valley_depth(zero, box)


def test_a_slanted_digit_stays_in_its_row():
    # most of the row is centred at y=100, but a sloping "1" hangs 50 px
    # lower - too far for the centre test, yet more than half of it overlaps
    row = [{"x": 0, "y": 40, "w": 40, "h": 120, "area": 900, "cy": 100.0},
           {"x": 60, "y": 90, "w": 40, "h": 120, "area": 900, "cy": 150.0}]
    assert len(split_rows(row, row_gap=50)) == 1


def test_ruled_line_is_found_but_digits_are_not():
    mask = np.zeros((300, 1000), np.uint8)
    cv2.line(mask, (20, 200), (980, 215), 255, 4)       # slightly tilted ruled line
    cv2.putText(mask, "38", (300, 180), cv2.FONT_HERSHEY_SIMPLEX, 4, 255, 6)
    lines = ruled_lines(mask)
    assert lines[207, 500] == 255                        # on the line
    assert not lines[60:170, 300:500].any()              # the digits are left alone


def test_crop_margin_stops_at_the_edge_of_the_paper():
    paper = np.zeros((200, 200), np.uint8)
    paper[:, 50:] = 255                                  # desk on the left
    x0, y0, x1, y1 = clip_to_paper(paper, (20, 20, 180, 180), (60, 60, 120, 120))
    assert x0 == 50 and (y0, x1, y1) == (20, 180, 180)


def test_paper_mask_excludes_a_coloured_desk():
    image = np.zeros((400, 400, 3), np.uint8)
    image[:] = (60, 140, 200)                       # light wood, BGR
    image[50:350, 100:300] = (235, 235, 235)        # white sheet
    mask = paper_mask(image, inset=10)
    assert mask[200, 200] == 255                    # middle of the sheet
    assert mask[200, 50] == 0                       # the desk
    assert mask[200, 102] == 0                      # within the inset of the edge
