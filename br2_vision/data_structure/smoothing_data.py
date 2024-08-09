import os
import h5py

from threading import Lock

from br2_vision.utility.logging import get_script_logger


class SmoothingData:
    _instance = None
    _lock = Lock()

    _key_time = "/smoothing/time"
    _key_data_index = "/smoothing/data_index"
    _key_radius = "/smoothing/radius"
    _key_position = "/smoothing/position"
    _key_director = "/smoothing/director"
    _key_shear = "/smoothing/shear"
    _key_kappa = "/smoothing/kappa"

    def __new__(cls, path):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SmoothingData, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, path):
        if self._initialized:
            return
        if not os.path.exists(path):
            raise FileNotFoundError(f"The file {path} does not exist.")
        self.path = path
        self.logger = get_script_logger(os.path.basename(__file__))
        self.file = None
        self._initialized = True

    def __enter__(self):
        self.file = h5py.File(self.path, "r+")

        # Create datasets if they don't exist
        if "time" not in self.file:
            self.file.create_dataset(self._key_time, (0,), dtype="f")
        if "data_index" not in self.file:
            self.file.create_dataset(self._key_data_index, (0,), dtype="i")
        if "radius" not in self.file:
            self.file.create_dataset(self._key_radius, (0,), dtype="f")
        if "position" not in self.file:
            self.file.create_dataset(self._key_position, (0, 3), dtype="f")
        if "director" not in self.file:
            self.file.create_dataset(self._key_director, (0, 3, 3), dtype="f")
        if "shear" not in self.file:
            self.file.create_dataset(self._key_shear, (0, 3), dtype="f")
        if "kappa" not in self.file:
            self.file.create_dataset(self._key_kappa, (0, 3), dtype="f")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
        self.file = None

    def get_time(self):
        return self.file[self._key_time][:]

    def get_data_index(self):
        return self.file[self._key_data_index][:]

    def get_radius(self):
        return self.file[self._key_radius][:]

    def get_position(self):
        return self.file[self._key_position][:]

    def get_director(self):
        return self.file[self._key_director][:]

    def get_shear(self):
        return self.file[self._key_shear][:]

    def get_kappa(self):
        return self.file[self._key_kappa][:]

    def set(
        self,
        time=None,
        data_index=None,
        radius=None,
        position=None,
        director=None,
        shear=None,
        kappa=None,
    ):
        if time is not None:
            if len(time) != len(self.file[self._key_time]):
                self.file[self._key_time].resize(len(time), axis=0)
            self.file[self._key_time][:] = time
        if data_index is not None:
            if len(data_index) != len(self.file[self._key_data_index]):
                self.file[self._key_data_index].resize(len(data_index), axis=0)
            self.file[self._key_data_index][:] = data_index
        if radius is not None:
            if len(radius) != len(self.file[self._key_radius]):
                self.file[self._key_radius].resize(len(radius), axis=0)
            self.file[self._key_radius][:] = radius
        if position is not None:
            if len(position) != len(self.file[self._key_position]):
                self.file[self._key_position].resize(len(position), axis=0)
            self.file[self._key_position][:] = position
        if director is not None:
            if len(director) != len(self.file[self._key_director]):
                self.file[self._key_director].resize(len(director), axis=0)
            self.file[self._key_director][:] = director
        if shear is not None:
            if len(shear) != len(self.file[self._key_shear]):
                self.file[self._key_shear].resize(len(shear), axis=0)
            self.file[self._key_shear][:] = shear
        if kappa is not None:
            if len(kappa) != len(self.file[self._key_kappa]):
                self.file[self._key_kappa].resize(len(kappa), axis=0)
            self.file[self._key_kappa][:] = kappa


# Example usage:
# with HDF5Singleton('path/to/your/file.h5') as obj:
#     obj.set(time=new_time_data, data_index=new_data_index_data)
#     time_data = obj.get_time()
