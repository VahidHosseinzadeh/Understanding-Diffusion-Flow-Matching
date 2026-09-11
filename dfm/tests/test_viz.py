"""Tests for the plotting helpers.

Only the logic is tested, not the pixels: frame selection has an
off-by-one that would silently drop the t=1 column, which is the one
frame anyone actually wants to look at.
"""
from __future__ import annotations

import torch

from viz import _frame_indices, save_trajectory_filmstrip, save_trajectory_filmstrip_2d


def test_frame_indices_span_the_whole_trajectory():
    """First and last frames must always appear -- t=0 is the noise you
    started from and t=1 is the sample, and a filmstrip missing either is
    not showing the thing it claims to."""
    idx = _frame_indices(n_points=51, n_frames=8)
    assert len(idx) == 8
    assert idx[0] == 0
    assert idx[-1] == 50
    assert idx == sorted(idx)


def test_frame_indices_are_evenly_spaced():
    idx = _frame_indices(n_points=41, n_frames=5)
    assert idx == [0, 10, 20, 30, 40]


def test_frame_indices_never_exceed_available_points():
    """Asking for more columns than the sampler took steps must not index
    out of range -- it just gives one column per available frame."""
    idx = _frame_indices(n_points=3, n_frames=10)
    assert len(idx) == 3
    assert idx == [0, 1, 2]


def test_filmstrip_writes_a_file(tmp_path):
    out = tmp_path / "nested" / "film.png"
    save_trajectory_filmstrip(torch.randn(21, 5, 1, 8, 8), out, n_frames=4, n_samples=3)
    assert out.exists() and out.stat().st_size > 0


def test_filmstrip_handles_fewer_samples_than_requested(tmp_path):
    out = tmp_path / "film.png"
    save_trajectory_filmstrip(torch.randn(11, 2, 1, 8, 8), out, n_frames=4, n_samples=8)
    assert out.exists()


def test_filmstrip_2d_writes_a_file(tmp_path):
    out = tmp_path / "film2d.png"
    save_trajectory_filmstrip_2d(torch.randn(21, 50, 2), out, n_frames=5,
                                 reference=torch.randn(100, 2))
    assert out.exists() and out.stat().st_size > 0
