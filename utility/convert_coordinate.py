import numpy as np
import scipy as sp
import scipy.linalg as spl
from collections import defaultdict

from sklearn.linear_model import LinearRegression

# Reference Marker Configuration (in m)
_THICKNESS = 0.002
_RADIUS = 0.006522 + _THICKNESS # Radius of tube
_CABLE_DRIVEN_RADIUS = 0.015

def rotate_point_around_point_2D(point, center, angle_ccw):
    # Rotate the coordinate in reference space to accomodate the center-offset
    s = np.sin(angle_ccw)
    c = np.cos(angle_ccw)

    point -= center
    point = np.array([point[0] * c - point[1] * s, point[0] * s + point[1] * c])
    point += center
    return point

def three_ring_xz_converter(loc_idx):
    """
    Given refernce point id, return reference coordinate in planar space.

    Parameters
    ----------
    loc_idx : str
    """

    # Geometry
    up = np.array([-_RADIUS, 0.0])
    left = np.array([0.0, -_RADIUS])
    right= np.array([0.0, _RADIUS])
    center = np.array([_RADIUS/np.cos(np.pi/6), 0])

    # Location
    if loc_idx == 0:
        pz = up
    elif loc_idx == 1:
        pz = left
    elif loc_idx == 2:
        pz = rotate_point_around_point_2D(right, center, 2*np.pi/3)
    elif loc_idx == 3:
        pz = rotate_point_around_point_2D(up, center, 2*np.pi/3)
    elif loc_idx == 4:
        pz = rotate_point_around_point_2D(left, center, 2*np.pi/3)
    elif loc_idx == 5:
        pz = rotate_point_around_point_2D(right, center, -2*np.pi/3)
    elif loc_idx == 6:
        pz = rotate_point_around_point_2D(up, center, -2*np.pi/3)
    elif loc_idx == 7:
        pz = rotate_point_around_point_2D(left, center, -2*np.pi/3)
    elif loc_idx == 8:
        pz = right
    else:
        raise NotImplementedError("using {}".format(loc_idx))
    return pz

def three_ring_xyz_converter(label, n_ring, ring_space):
    """
    Three-ring cross-sectional plane on BR2 (07/05/21)
    """
    vec = np.zeros([3])
    cumulative_space = np.cumsum(ring_space)

    # Determine x, z
    base_char = label[0]; label = label[1:]
    if base_char == 'R':
        y_idx, loc_idx = label.split('-')
        y_idx = int(y_idx) - 1
        loc_idx = int(loc_idx)
        # y
        vec[1] = cumulative_space[y_idx]
        pz = three_ring_xz_converter(loc_idx)
    else:
        raise NotImplementedError
    vec[0] = pz[0]
    vec[2] = pz[1]

    return vec

def get_center_and_normal(labelled_points:dict, n_ring:int):
    # Return Placeholder
    center_position = np.zeros([3, n_ring])
    director_vector = np.zeros([3, 3, n_ring])

    # Parsing
    point_group = defaultdict(dict)
    for label, point in labelled_points.items():
        # Parsing Label
        base_char = label[0]; label = label[1:]
        if base_char == 'S':
            continue
        y_idx, loc_idx = label.split('-')
        y_idx = int(y_idx) - 1
        loc_idx = int(loc_idx)

        point_group[y_idx][loc_idx] = point
    
    tangent_offset_angle = []
    for i in range(n_ring):
        points = point_group[i]
        # Get Plane and Normal Vector
        A = []
        for _, point in points.items():
            A.append(point)
        assert len(A) > 1, f"Each plane must have more than 1 reference point to interpolate properly. {point_group}"
        A = np.array(A)
        y = np.ones([A.shape[0]])
        tangent = spl.pinv2(A)@y
        tangent /= np.linalg.norm(tangent) # tangentize
        # Check direction or tangent vector
        if i == 0: # First vector should be in positive y-direction
            if tangent[1] < 0:
                tangent *= -1
        else: # Following vector should be "least deviated" from previous vector
            if np.linalg.norm(tangent-prev_tangent) > np.linalg.norm(-tangent-prev_tangent):
                tangent *= -1
        prev_tangent = tangent
        director_vector[2,:,i] = tangent

        # Get Center Position
        # Geometry
        plane_center = np.array([_RADIUS/np.cos(np.pi/6), 0.0, 0.0])  # Point we want to find in local frame
        plane_space = []
        dlt_space = []
        for loc_idx, point in points.items():
            vec = np.zeros([3]) 
            vec[[0,2]] = three_ring_xz_converter(loc_idx)

            plane_space.append(vec)
            dlt_space.append(point)
        if len(plane_space) <= 4:
            print(f'WARNING : Interpolation with {len(plane_space)} point: {n_ring=}')
        reg = LinearRegression(fit_intercept=True, normalize=False).fit(plane_space, dlt_space)
        center_position[:,i] = reg.predict([plane_center])[0]

        # Get normal and binormal
        plane_rod1_center = np.array([0.0, 0.0, 0.0])
        plane_normal = np.array([0.0, 0.0, 1.0])
        plane_binormal = np.array([1.0, 0.0, 0.0])
        res = reg.predict([plane_rod1_center, plane_normal, plane_binormal])
        normal = res[1]-res[0]
        binormal = res[2]-res[0] # interpolated binormal
        binormal2 = np.cross(tangent, normal)
        normal /= np.linalg.norm(normal)
        binormal /= np.linalg.norm(binormal)
        binormal2 /= np.linalg.norm(binormal2)
        theta = np.arccos(np.dot(binormal, binormal2))
        tangent_offset_angle.append(theta)
        director_vector[0,:,i] = normal
        director_vector[1,:,i] = binormal
        director_vector[2,:,i] = np.cross(normal, binormal)
        #director_vector[1,:,i] = np.cross(director_vector[2,:,i], normal) # interpolated binormal
    return center_position, director_vector


def append_all_marker_points(labelled_points:dict, num_ring:int):
    """
    Given partial labelled point information, infer all marker points.
    The method is built for the concurrent optical flow + DLT algorithm.
    The method assume all markers are recognized at all time.
    """
    return_points = {}
    return_points['S0'] = labelled_points['S0']
    return_points['S1'] = labelled_points['S1']
    return_points['S2'] = labelled_points['S2']
    num_markers = 9
    for ring_idx in range(1, num_ring+1):
        available_point_labels = [key for key in labelled_points.keys() if f'R{ring_idx}' in key]

        # Find Mapper
        plane_space = [] # 2d xz plane point
        dlt_space = [] # xyz point given by DLT
        for label in available_point_labels:
            loc_idx = int(label.split('-')[1])
            xz  = three_ring_xz_converter(loc_idx)

            plane_space.append(xz)
            dlt_space.append(labelled_points[label])
        reg = LinearRegression(fit_intercept=True, normalize=False).fit(plane_space, dlt_space)

        # Map all markers
        labels = []
        xzs = []
        for marker_idx in range(0, num_markers):
            new_label = f'R{ring_idx}-{marker_idx}' 
            if new_label in available_point_labels: # already existing point
                return_points[new_label] = labelled_points[new_label]
                continue
            xzs.append(three_ring_xz_converter(marker_idx))
            labels.append(new_label)
        if len(xzs) > 0 :
            for tag, value in zip(labels, reg.predict(np.array(xzs))):
                return_points[tag] = np.array(value) # Update labelled_points
    return return_points
