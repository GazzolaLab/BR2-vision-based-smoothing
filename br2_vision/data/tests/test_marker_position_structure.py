import pytest

import os, sys
import numpy as np
import h5py
import yaml
import pathlib
from br2_vision.data import MarkerPositions


class TestVariables:
    center_offset = [1., 2., 3.]
    marker_positions = {'A': (0., 0., 1.), 'B': (1., 0., 0.)}


class TestMarkerPositions(TestVariables):

    @pytest.fixture(scope="session")
    def mock_marker_position(self):
        mp = MarkerPositions(
            marker_center_offset=self.center_offset,
            marker_positions=self.marker_positions,
        )
        return mp

    # fixture to create test yaml file im temp_dir
    @pytest.fixture(scope="session")
    def position_yaml(self, tmp_path_factory, mock_marker_position):
        filename = tmp_path_factory.mktemp("position_data") / "markers.yaml"
        mock_marker_position.to_yaml(filename)
        return filename

    def test_marker_position_creation_from_yaml(self, mock_marker_position):
        assert mock_marker_position.marker_center_offset == self.center_offset
        assert mock_marker_position.marker_positions['A'] == self.marker_positions['A'] 
        assert mock_marker_position.marker_positions['B'] == self.marker_positions['B']

        
    def test_init(self):
        mp = MarkerPositions(
            marker_center_offset=self.center_offset,
            marker_positions=self.marker_positions,
        )
        assert mp.marker_center_offset == self.center_offset
        assert mp.marker_positions == self.marker_positions

    def test_to_yaml(self, tmp_path, mock_marker_position):
        filename = tmp_path / "markers_test.yaml"
        mock_marker_position.to_yaml(filename)
        assert filename.exists()

        loaded_marker_position = MarkerPositions.from_yaml(file_path=filename)
        assert mock_marker_position == loaded_marker_position
        assert mock_marker_position.marker_center_offset == self.center_offset
        assert mock_marker_position.marker_positions == self.marker_positions

    def test_keys(self, mock_marker_position):
        assert mock_marker_position.tags == list(self.marker_positions.keys())

    def test_length(self, mock_marker_position):
        assert len(mock_marker_position) == len(list(self.marker_positions.keys()))

    def test_to_hdf5(self, tmp_path, mock_marker_position):
        filename = tmp_path / "markers.h5"
        mock_marker_position.to_h5(filename)
        assert filename.exists()

    def test_from_hdf5(self, tmp_path, mock_marker_position):
        filename = tmp_path / "markers.h5"
        mock_marker_position.to_h5(filename)
        mp2 = MarkerPositions.from_h5(filename)
        assert mp2 == mock_marker_position
        np.testing.assert_allclose(mp2.marker_center_offset, self.center_offset)
        assert mp2.tags == mock_marker_position.tags
        assert mp2.marker_positions == self.marker_positions

    def test_get_total_count(self, mock_marker_position):
        assert mock_marker_position.get_total_count() == len(mock_marker_position) * len(mock_marker_position.marker_center_offset)

    def test_get_count_per_ring(self, mock_marker_position):
        assert mock_marker_position.get_count_per_ring() == len(mock_marker_position.marker_center_offset)

    @pytest.mark.parametrize("zid, tag", [(0, "A"), (0, "A"), (1,"B"), (2,"B")])
    def test_get_position(self, mock_marker_position, zid, tag):
        z_level = np.cumsum(self.center_offset)
        Q = np.array([[0, 1, 0],
                      [0, 0, 1],
                      [1, 0, 0]]).astype(np.float_)
        np.testing.assert_allclose(Q, mock_marker_position.Q)

        expected_value = z_level[zid] * Q[:,2] + Q@self.marker_positions[tag]
        return_value = mock_marker_position.get_position(zid, tag)
        np.testing.assert_allclose(return_value, expected_value)

    def test_Q_composition(self):
        mp = MarkerPositions(
            marker_center_offset=self.center_offset,
            marker_positions=self.marker_positions,
            marker_direction=(1.0, 0.0, 0.0),
            normal_direction=(0.0, 1.0, 0.0),
        )
        Q_expected = np.array([[0, 0, 1],
                               [1, 0, 0],
                               [0, 1, 0]]).astype(np.float_)
        np.testing.assert_allclose(mp.Q, Q_expected)
