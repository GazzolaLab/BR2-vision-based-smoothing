"""Scikit-learn style API for forward-backward rod smoothing."""

from __future__ import annotations
from typing import Sequence, Union

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from elastica.rod.cosserat_rod import CosseratRod

from br2_vision.algorithms.rod_tools import shear_and_curvature_to_strain
from br2_vision.algorithms.smoothing_algorithm import ForwardBackwardSmooth

SparseInput = Union["SparseFrame", "SparseSequence"]


@dataclass
class SparseFrame:
    """Sparse position and orientation at one timestep."""

    position: np.ndarray  # (3, n_markers)
    director: np.ndarray  # (3, 3, n_markers)

    def __post_init__(self) -> None:
        position = np.asarray(self.position)
        director = np.asarray(self.director)
        if position.ndim != 2 or position.shape[0] != 3 or position.shape[1] < 1:
            raise ValueError(
                "SparseFrame.position must have shape (3, n_markers) with n_markers >= 1, "
                f"got {position.shape}."
            )
        n_markers = position.shape[1]
        if director.shape != (3, 3, n_markers):
            raise ValueError(
                f"SparseFrame.director must have shape (3, 3, {n_markers}), "
                f"got {director.shape}."
            )

    @property
    def n_markers(self) -> int:
        return self.position.shape[1]


@dataclass
class SparseSequence:
    """Sparse position and orientation over time."""

    time: np.ndarray  # (n_frames,)
    position: np.ndarray  # (n_frames, 3, n_markers)
    director: np.ndarray  # (n_frames, 3, 3, n_markers)

    def __post_init__(self) -> None:
        time = np.asarray(self.time)
        position = np.asarray(self.position)
        director = np.asarray(self.director)

        if time.ndim != 1:
            raise ValueError(f"SparseSequence.time must be 1D, got shape {time.shape}.")
        if position.ndim != 3 or position.shape[1] != 3 or position.shape[2] < 1:
            raise ValueError(
                "SparseSequence.position must have shape (n_frames, 3, n_markers) "
                f"with n_markers >= 1, got {position.shape}."
            )
        if director.ndim != 4 or director.shape[1:3] != (3, 3):
            raise ValueError(
                "SparseSequence.director must have shape (n_frames, 3, 3, n_markers), "
                f"got {director.shape}."
            )
        if time.shape[0] != position.shape[0] or time.shape[0] != director.shape[0]:
            raise ValueError(
                "SparseSequence time, position, and director lengths must match."
            )

        n_markers = position.shape[2]
        if director.shape[3] != n_markers:
            raise ValueError(
                "SparseSequence position and director marker counts must match."
            )

    @property
    def n_markers(self) -> int:
        return self.position.shape[2]


@dataclass
class DenseFrame:
    """Dense reconstructed rod state at one timestep."""

    position: np.ndarray  # (3, n_nodes)
    director: np.ndarray  # (3, 3, n_nodes)
    shear: np.ndarray  # (3, n_elems)
    kappa: np.ndarray  # (3, n_elems - 1)
    strain: np.ndarray  # packed strain vector
    cost: float
    n_iter: int


def _normalized_marker_arc_lengths(
    marker_center_offset: Sequence[float],
) -> tuple[np.ndarray, float]:
    """Return normalized marker arc lengths and total rod length."""
    s_position = np.cumsum(np.asarray(marker_center_offset, dtype=float))
    rod_length = float(s_position[-1])
    return s_position / rod_length, rod_length


def _create_rod(
    n_elems: int,
    rod_length: float,
    radius: float,
    direction: np.ndarray,
    normal: np.ndarray,
    youngs_modulus: float,
) -> CosseratRod:
    radii = radius * np.ones(n_elems + 1)
    radii_mean = (radii[:-1] + radii[1:]) / 2
    return CosseratRod.straight_rod(
        n_elements=n_elems,
        start=np.zeros((3,)),
        direction=direction,
        normal=normal,
        base_length=rod_length,
        base_radius=radii_mean.copy(),
        density=700,
        youngs_modulus=youngs_modulus,
    )


class ForwardBackwardSmoother(BaseEstimator, TransformerMixin):
    """Fit rod geometry once, then reconstruct dense states from sparse poses."""

    def __init__(
        self,
        marker_center_offset: Sequence[float],
        *,
        n_elems: int = 100,
        radius: float = 0.0075 * 2.742,
        direction: Sequence[float] = (0.0, 1.0, 0.0),
        normal: Sequence[float] = (1.0, 0.0, 0.0),
        youngs_modulus: float = 1e7,
        fit_director: bool = True,
        argument_weight: float = 1.0,
        step_size: float = 1e-6,
        data_weight: float = 1_000_000.0,
        max_iter: int = 100_000,
        tol: float = 1e-5,
        warm_start: bool = True,
    ) -> None:
        self.marker_center_offset = marker_center_offset
        self.n_elems = n_elems
        self.radius = radius
        self.direction = direction
        self.normal = normal
        self.youngs_modulus = youngs_modulus
        self.fit_director = fit_director
        self.argument_weight = argument_weight
        self.step_size = step_size
        self.data_weight = data_weight
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start

    def fit(
        self,
        X: SparseInput,
        y=None,
    ) -> "ForwardBackwardSmoother":
        reference = self._coerce_reference(X)
        s_position, rod_length = _normalized_marker_arc_lengths(self.marker_center_offset)
        n_markers = reference.n_markers

        if n_markers != len(self.marker_center_offset):
            raise ValueError(
                f"marker_center_offset length ({len(self.marker_center_offset)}) must match "
                f"number of markers in sparse data ({n_markers})."
            )

        direction = np.asarray(self.direction, dtype=float)
        normal = np.asarray(self.normal, dtype=float)
        rod = _create_rod(
            n_elems=self.n_elems,
            rod_length=rod_length,
            radius=self.radius,
            direction=direction,
            normal=normal,
            youngs_modulus=self.youngs_modulus,
        )

        if isinstance(reference, SparseFrame):
            noisy_position = reference.position
            noisy_director = reference.director
        else:
            noisy_position = reference.position[0]
            noisy_director = reference.director[0]

        data = self._build_algorithm_data(
            noisy_position=noisy_position,
            noisy_director=noisy_director,
            s_position=s_position,
            s_director=s_position.copy(),
        )
        self.algo_ = ForwardBackwardSmooth(rod, self._algorithm_config(), data)
        self.n_markers_ = n_markers
        self.rod_length_ = rod_length
        self.s_position_ = s_position
        self.s_director_ = s_position.copy()
        self.fitted_ = True
        return self

    def transform(
        self,
        X: SparseInput,
    ) -> DenseFrame | list[DenseFrame]:
        check_is_fitted(self, "fitted_")

        if isinstance(X, SparseFrame):
            return self._transform_one(X)

        outputs = []
        for frame_index in range(X.time.shape[0]):
            frame = SparseFrame(
                position=X.position[frame_index],
                director=X.director[frame_index],
            )
            outputs.append(self._transform_one(frame))
        return outputs

    def fit_transform(
        self,
        X: SparseInput,
        y=None,
    ) -> DenseFrame | list[DenseFrame]:
        return self.fit(X, y).transform(X)

    def _transform_one(self, frame: SparseFrame) -> DenseFrame:
        if frame.n_markers != self.n_markers_:
            raise ValueError(
                f"Expected {self.n_markers_} markers, got {frame.n_markers}."
            )
        self.algo_.data.noisy_position = np.asarray(frame.position, dtype=float)
        self.algo_.data.noisy_director = np.asarray(frame.director, dtype=float)

        cost, n_iter = self.algo_.run(
            iter_number=self.max_iter,
            threshold=self.tol,
            cost_threshold=self.data_weight * 10,
        )
        dense = DenseFrame(
            position=self.algo_.position.copy(),
            director=self.algo_.director.copy(),
            shear=self.algo_.shear.copy(),
            kappa=self.algo_.kappa.copy(),
            strain=shear_and_curvature_to_strain(self.algo_.shear, self.algo_.kappa),
            cost=float(cost[-1]),
            n_iter=int(n_iter),
        )

        if not self.warm_start:
            self.fit(frame)
        return dense

    @staticmethod
    def _coerce_reference(X: SparseInput) -> SparseInput:
        if isinstance(X, (SparseFrame, SparseSequence)):
            return X
        raise TypeError(
            "X must be a SparseFrame or SparseSequence, "
            f"got {type(X).__name__}."
        )

    def _algorithm_config(self) -> SimpleNamespace:
        config = SimpleNamespace()
        config.argument_weight = self.argument_weight
        config.step_size = self.step_size
        config.data_deviation_weight_cost = self.data_weight
        config.data_deviation_weight_cost_position = self.data_weight
        config.data_deviation_weight_cost_director = self.data_weight
        return config

    def _build_algorithm_data(
        self,
        noisy_position: np.ndarray,
        noisy_director: np.ndarray,
        s_position: np.ndarray,
        s_director: np.ndarray,
    ) -> SimpleNamespace:
        data = SimpleNamespace()
        data.noisy_position = noisy_position
        data.noisy_director = noisy_director
        data.s_position = s_position
        data.s_director = s_director
        data.director_flag = self.fit_director
        return data
