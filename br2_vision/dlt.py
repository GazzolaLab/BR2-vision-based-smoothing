import os
import sys

import json

import numpy as np
import numpy.linalg as la

import scipy.stats as ss
import scipy.linalg as spl

# http://www.kwon3d.com/theory/dlt/dlt.html

"""
DLT module
"""


class DLT:
    """
    End-to-end Direct Linear Transformation (DLT) module.
    Provide calibration, interpolation, and inverse-interpolation features.
    The module itself can be saved and loaded for future usage.
    """
    def __init__(self, calibration_path=None):
        """__init__.

        Parameters
        ----------
        calibration_path : str
            Save path for the calibration data. The directory should include
            sub-directories for each camera calibration.
        """
        self._finalize = False

        self.num_camera = 0
        self.camera = {} 
        self.camera_desc = {}  # Camera description
        self.camera_param = {} # Camera parameters

        self.distortion_correction_threshold = 1e-4  # 0.1 mm distortion allowance

        self.calibration_path = calibration_path
        if calibration_path is not None:
            os.makedirs(self.calibration_path, exist_ok=True)

    def map(self, uvs, distortion_correction=False, return_error=False):
        """
        DLT map: image-space(uv) to object-space(xyz)
        Reconstruction

        Parameters
        ----------
        uvs : dict
            Dictionary of image-space coordinate for each camera
        distortion_correction : bool
            Use distortion correction (iterative method) to find xyz coordinate
        Returns
        -------
        xyz : array
            3d object-space coordinate
        """
        # Build Q and q TODO: Revisit. Account for noise
        Q, q = [], []
        for k, (u,v) in uvs.items():
            param = self.camera_param[k]
            # Build matrix
            Q.append([param[0]-param[8]*u, param[1]-param[9]*u, param[2]-param[10]*u])
            Q.append([param[4]-param[8]*v, param[5]-param[9]*v, param[6]-param[10]*v])
            q.append(u-param[3])
            q.append(v-param[7])
        Q = np.asarray(Q)
        q = np.asarray(q)

        # Moore-Penrose pseudo-inverse method
        assert Q.shape[0] >= 4, "Least 2 camera-coordinate must be given"
        xyz = spl.pinv(Q)@q
        condition_number = np.linalg.cond(Q)

        # Distortion Correction
        if distortion_correction:
            xyz_corrected = xyz.copy()  # Initial guess
            delta = 1
            while delta > self.distortion_correction_threshold:
                Q, q = [], []
                for k, (u,v) in uvs.items():
                    param = self.camera_param[k]
                    u0, v0 = self.camera[k].info['camera position']
                    # Distortion correction
                    xi, eta = u-u0, v-v0 
                    r_sqr = xi**2 + eta**2  # eqn 12
                    du = xi*(param[11]*r_sqr + param[12]*r_sqr**2 + param[13]*r_sqr**3) + \
                        param[14]*(r_sqr+2*xi**2) + param[15]*xi*eta
                    dv = eta*(param[11]*r_sqr + param[12]*r_sqr**2 + param[13]*r_sqr**3) + \
                        param[15]*(r_sqr+2*eta**2) + param[14]*xi*eta
                    u = u-du
                    v = v-dv
                    R = param[8]*xyz[0] + param[9]*xyz[1] + param[10]*xyz[2] + 1
                    # Build matrix
                    Q.append([(param[0]-param[8]*u)/R, (param[1]-param[9]*u)/R, (param[2]-param[10]*u)/R])
                    Q.append([(param[4]-param[8]*v)/R, (param[5]-param[9]*v)/R, (param[6]-param[10]*v)/R])
                    q.append((u-param[3])/R)
                    q.append((v-param[7])/R)
                Q = np.asarray(Q)
                q = np.asarray(q)

                # Moore-Penrose pseudo-inverse method
                xyz_corrected_next = spl.pinv(Q)@q
                delta = spl.norm(xyz_corrected_next-xyz_corrected, np.inf)
                xyz_corrected = xyz_corrected_next
            xyz = xyz_corrected

        # Compute Error
        if return_error:
            error_uv = (Q@xyz + q) / np.linalg.norm(Q, axis=1)
            error = np.sqrt(error_uv[::2]**2 + error_uv[1::2]**2)
            error_dict = dict([(a, b) for a, b in zip(uvs.keys(), error)])
            return xyz, condition_number, error_dict
        else:
            return xyz, condition_number

    def map_error(self, uvs, xyz, single_value=True):
        """map_error.

        Calculate the distance between the extimated point (xyz) to each camera seeing (uvs) points.

        Parameters
        ----------
        uvs :
            uvs
        xyz :
            xyz
        """
        # Build Q and q TODO: Revisit. Account for noise
        Q, q = [], []
        for k, (u,v) in uvs.items():
            param = self.camera_param[k]
            # Build matrix
            Q.append([param[0]-param[8]*u, param[1]-param[9]*u, param[2]-param[10]*u])
            Q.append([param[4]-param[8]*v, param[5]-param[9]*v, param[6]-param[10]*v])
            q.append(u-param[3])
            q.append(v-param[7])
        Q = np.asarray(Q)
        q = np.asarray(q)

        if single_value:
            error_uv = (Q@xyz + q) / np.linalg.norm(Q, axis=1)
            error = np.sqrt(error_uv[::2]**2 + error_uv[1::2]**2)
            error_dict = dict([(a, b) for a, b in zip(uvs.keys(), error)])
            return error_dict
        else:
            error_uv = np.abs(Q@xyz + q) / np.linalg.norm(Q, axis=1)
            error = list(zip(error_uv[::2], error_uv[1::2]))
            error_dict = dict([(a, b) for a, b in zip(uvs.keys(), error)])
            return error_dict

    def inverse_map(self, x, y, z):
        """inverse_map.
        DLT inverse map: object-space(xyz) to image-space(uv)

        Parameters
        ----------
        x,y,z : float
            Coordinate in ob
        Return
        ------
        uvs : dict
            Dictionary of image-space coordinate for each camera
        """
        uvs = {}
        for k, param in self.camera_param.items():
            u = (param[0]*x+param[1]*y+param[2]*z+param[3]) / (param[8]*x+param[9]*y+param[10]*z+1)
            v = (param[4]*x+param[5]*y+param[6]*z+param[7]) / (param[8]*x+param[9]*y+param[10]*z+1)
            uvs[k] = (u,v)
        return uvs


    def save(self, force=False):
        """ Save the calibration
        In default, save only when the calibration process can be completed.
        If want to save before DLT calibration is done, use 'force' parameter.

        Save parameter in numpy npz format.
        Save camera information in json format

        Parameters
        ----------
        force : bool
            Save even the calibration is not done
        """
        if not self._finalize and not force:
            print('Save incomplete')
            return 0
        if self.calibration_path is None:
            print('Calibration path not determined')
            return 0
        save_path_information = os.path.join(self.calibration_path, 'camera_information.json')
        save_path_parameters = os.path.join(self.calibration_path, 'camera_parameters.npz')

        # Save camera information in JSON
        json_dict = {}
        for k, camera in self.camera.items():
            json_dict[k] = camera.info
        with open(save_path_information, 'w') as file:
            json.dump(json_dict, file)

        # Save camera parameters in NPZ
        save_dictionary = {str(k):np.array(v) for k, v in self.camera_param.items()}
        np.savez(
            save_path_parameters,
            **save_dictionary
        )
        print('Save completed')
        return 1

    def load(self):
        """ 
        Load previous calibration parameters.
        TODO: num camera
        """
        if self.calibration_path is None:
            print('Calibration path not determined')
            return 0

        data = np.load(os.path.join(self.calibration_path, 'camera_parameters.npz'))
        self.camera_param = {}
        for k, v in data.items():
            if len(v) != 16:
                raise ValueError('loaded data does not have correct camera parameters')
            self.camera_param[int(k)] = np.asarray(v)
        print('DLT: Load completed')
        return 1

    def add_camera(self, camera_id: int, calibration_type=16, tag='', verbose=False):
        """ Add new camera into the system

        Parameters
        ----------
        camera_id : int
            Camera id number
        calibration_type : int
            Number of parameter use for calibration (default=16)
            Support: 11, 12, 14, 16
        tag : str
            Add camera description if necessary (default='')
            It is for the purpose of documentation.
        verbose : bool
            Debugging parameter to print the progress (default=False)
        """
        self.camera[camera_id] = self.Camera(calibration_type=calibration_type)
        self.camera_desc[camera_id] = tag
        self.num_camera += 1
        if verbose:
            print('New Camera Added - id: {},  type: {}, desc: {}'
                    .format(camera_id, calibration_type, tag))

    def add_reference(self, u, v, x, y, z, camera_id: int):
        """
        Add reference point to designated camera
        """
        assert camera_id in self.camera, f"Camera {camera_id} does not exist"
        self.camera[camera_id].references.append([u, v, x, y, z])

    def finalize(self, verbose=False):
        # Run calibration for each camera and save the parameters
        for k, camera in self.camera.items():
            param = camera.calculate_parameters(verbose=verbose)
            self.camera_param[k] = param

            error, error_vector = camera.calibration_error()
            error_vector = np.array(error_vector)
            if verbose:
                print('camera {} calibration error (in pixel) : {}+={}'.format(k, error, error_vector.std()))
        self._finalize = True

    def reconstruction_error(self, reference_points):
        # Reconstruction Error (eqn 36)
        pass

    class Camera:
        def __init__(self, calibration_type=16):
            assert calibration_type in [11, 12, 14, 16]
            self.type = calibration_type
            self.references = []
            self.info = {}  # json compatible dictionary!!

            self.iterative_threshold = 1e-4

        def calculate_parameters(self, verbose=False):
            """
            Compute the camera paramters and return
            """
            assert self.validity(), 'Not enough reference points are given to calibrate the camera'

            param = np.zeros(16)  # Calibration Parameter

            # DLT Calibration Parameters (1~11)
            L = []
            g = []
            for u, v, x, y, z in self.references:
                L.append([x,y,z,1,0,0,0,0,-u*x,-u*y,-u*z])
                L.append([0,0,0,0,x,y,z,1,-v*x,-v*y,-v*z])
                g.append(u)
                g.append(v)
            param[:11] = spl.pinv(L) @ g

            # Camera Position (eqn 25)
            A = spl.inv(np.stack([param[0:3], param[4:7], param[8:11]]))
            b = np.array([-param[3], -param[7], -1])
            camera_position = A @ b
            self.info['camera position'] = camera_position.tolist()

            # Principal Point (u0, v0) (eqn 26, 27)
            Dsquare = 1.0/(np.dot(param[8:11], param[8:11]))
            u0 = Dsquare * np.dot(param[0:3], param[8:11])
            v0 = Dsquare * np.dot(param[5:8], param[8:11])
            self.info['principal point'] = (u0, v0)

            # 3D Distortion Terms (12-16) (eqn 12, 14, 16)
            if self.type > 11:
                # Initialize extended parameter
                param_ext = np.zeros(self.type); param_ext[:11] = param[:11]
                # Precompute xi, eta, and r^2
                for idx, (u, v, x, y, z) in enumerate(self.references):
                    xi, eta = u-u0, v-v0 
                    r_sqr = xi**2 + eta**2  # eqn 12
                    self.references[idx].extend([xi, eta, r_sqr])
                # Iterative method (eqn 16)
                delta = 1
                iteration = 0
                while delta > self.iterative_threshold:
                    iteration += 1
                    # Define extended matrix with distortion information
                    L_ext = np.zeros([len(self.references)*2, self.type])
                    L_ext[:,:11] = np.array(L)
                    g_ext = g.copy()
                    for idx, (u, v, x, y, z, xi, eta, r_sqr) in enumerate(self.references):
                        R = param_ext[8]*x + param_ext[9]*y + param_ext[10]*z + 1  # eqn 14
                        print(xi, eta, r_sqr)
                        if self.type == 12:  # 3rd order optical distortion
                            L_ext[2*idx  , 11] = xi * R * r_sqr
                            L_ext[2*idx+1, 11] = eta * R * r_sqr
                        elif self.type == 14:  # 3rd order OD + decentering distortion
                            L_ext[2*idx  , 11] = xi * R * r_sqr
                            L_ext[2*idx+1, 11] = eta * R * r_sqr
                            L_ext[2*idx  , 12] = (2*xi**2 + r_sqr) * R
                            L_ext[2*idx+1, 12] = xi * eta * R
                            L_ext[2*idx  , 13] = eta * xi * R
                            L_ext[2*idx+1, 13] = (2*eta**2 + r_sqr) * R
                        elif self.type == 16:  # 3rd, 5th, 7th order OD + decentering distortion
                            L_ext[2*idx  , 11] = xi * R * r_sqr
                            L_ext[2*idx+1, 11] = eta * R * r_sqr
                            L_ext[2*idx  , 12] = xi * R * (r_sqr ** 2)
                            L_ext[2*idx+1, 12] = eta * R * (r_sqr ** 2)
                            L_ext[2*idx  , 13] = xi * R * (r_sqr ** 3)
                            L_ext[2*idx+1, 13] = eta * R * (r_sqr ** 3)
                            L_ext[2*idx  , 14] = (2*xi**2 + r_sqr) * R
                            L_ext[2*idx+1, 14] = xi * eta * R
                            L_ext[2*idx  , 15] = eta * xi * R
                            L_ext[2*idx+1, 15] = (2*eta**2 + r_sqr) * R
                        else:
                            raise NotImplementedError('Only 11,12,14,16 calibration type is implemented')
                        L_ext[2*idx:2*idx+2,:] /= R
                        g_ext[2:idx:2*idx+2] /= R
                    # Calculate difference between iteration
                    #param_ext_next = spl.pinv(L_ext) @ g
                    param_ext_next = jacobi_step(L_ext, g, param_ext)
                    delta = spl.norm(param_ext_next - param_ext, np.inf)
                    error = spl.norm(L_ext @ param_ext_next - g)
                    param_ext = param_ext_next
                    if verbose:
                        print('Iteration {}: delta={}, target threshold={}, error={}'
                                .format(iteration, delta, self.iterative_threshold, error))
                # Save modified parameters
                self.info['DLT parameters (undistorted)'] = param.copy().tolist()
                if self.type == 12:
                    param[:12] = param_ext
                elif self.type == 16:
                    param[:16] = param_ext
                elif self.type == 14:
                    param[:12] = param_ext[:12]
                    param[-2:] = param_ext[-2:]

            self.info['DLT parameters'] = param.tolist()
            return param

        def validity(self):
            """ 
            Check validity of the camera calibration:
                - Assure the enough number of reference points are given
            """
            num_point = len(self.references)
            return num_point >= self.type

        def calibration_error(self):
            """ eqn 35 """
            L = self.info['DLT parameters']
            ec = []
            for u, v, x, y, z in self.references:
                eps_u = u - (L[0]*x + L[1]*y + L[2]*z + L[3])/(L[8]*x + L[9]*y + L[10]*z + 1)
                eps_v = v - (L[4]*x + L[5]*y + L[6]*z + L[7])/(L[8]*x + L[9]*y + L[10]*z + 1)
                ec.append(np.sqrt(eps_u**2 + eps_v**2))
            self.info['calibration error'] = np.mean(ec)
            return np.mean(ec), ec


class DLT2D:
    """
    End-to-end Direct Linear Transformation (DLT) module for planar calibration.
    Provide calibration, interpolation, and inverse-interpolation features.
    The module itself can be saved and loaded for future usage.
    
    This module is specifically used for the multi-frame DLT calibration.
    Each frame is calibrated with 2D-DLT.
    The methods are instincts and bound to single camera.
    """
    def __init__(self, save_path='calibration.npz'):
        # Configuration
        self.save_path = save_path
        self._finalize = False

        # Algorithm Parameters
        self.param = np.zeros(8, dtype=float)

        # Data Placeholder
        self.references = []

    def summary(self):
        pass

    def clear(self):
        self._finalize = False
        self.references = []

    def map(self, u, v):
        """
        DLT map: image-space(uv) to object-space(xy)
        Reconstruction
        """
        raise NotImplementedError("Co-planar mapping is not possible with single camera.")

    def inverse_map(self, x, y):
        """inverse_map.
        DLT inverse map: object-space(xy) to image-space(uv)

        Parameters
        ----------
        x,y,z : float
            Coordinate in object space
        Return
        ------
        u,v : int
            Image-space coordinate
        """
        param = self.param
        u = int((param[0]*x+param[1]*y+param[2]) / (param[6]*x+param[7]*y+1))
        v = int((param[3]*x+param[4]*y+param[5]) / (param[6]*x+param[7]*y+1))
        return u, v 

    def save(self, force=False):
        """ Save the calibration
        In default, save only when the calibration process can be completed.
        If want to save before DLT calibration is done, use 'force' parameter.

        Save parameter in numpy npz format.

        Parameters
        ----------
        force : bool
            Save even the calibration is not done
        """
        if not self._finalize and not force:
            print('Save incomplete')
            return 0

        # Save camera parameters in NPZ
        np.savez(
            self.save_path,
            param=self.param
        )
        print('Save completed')
        return 1

    def load(self, pass_if_does_not_exist):
        """ 
        Load previous calibration parameters.
        """
        if pass_if_does_not_exist:
            if not os.path.exists(self.save_path):
                return
        data = np.load(self.save_path)
        self.param = data['param']
        print('DLT: Load completed')
        return 1

    def add_reference(self, u, v, x, y):
        """
        Add reference point
        (u,v) - image space 
        (x,y) - object space
        """
        self.references.append([u, v, x, y])

    def finalize(self, verbose=False):
        # Run calibration save the parameters
        self.param = self.calculate_parameters(verbose=verbose)
        self._finalize = True

    def calculate_parameters(self, verbose=False):
        """
        Compute the camera paramters and return
        """
        assert self.validity(), 'Not enough reference points are given to calibrate the camera'

        param = np.zeros(8)  # Calibration Parameter

        # DLT Calibration Parameters (32)
        L = []
        g = []
        for u, v, x, y in self.references:
            L.append([x,y,1,0,0,0,-u*x,-u*y])
            L.append([0,0,0,x,y,1,-v*x,-v*y])
            g.append(u)
            g.append(v)
        return spl.pinv(L) @ g

    def validity(self):
        """ 
        Check validity of the camera calibration:
            - Assure the enough number of reference points are given
            - (Left for more validity check)
        """
        return len(self.references) >= 4


def jacobi_step(A, b, x):
    xk = np.zeros_like(x)
    p,l,u = spl.lu(A)
    pinv = spl.inv(p)
    for i in range(xk.shape[0]):
        if A[i,i] == 0:
            print(i)
        xk[i] = (-(np.dot(A[i,:], x) - (A[i,i] * x[i])) + b[i]) / A[i,i]
    return xk
