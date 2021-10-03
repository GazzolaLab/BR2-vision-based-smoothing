import os
import sys
import numpy as np
from sklearn.linear_model import LinearRegression
from itertools import product, combinations

from collections import defaultdict

from dlt import DLT

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import matplotlib.cm as cm

from utility.convert_coordinate import three_ring_xyz_converter

# CONFIGURATION
EXCLUDE_TAGS = []
color_scheme = plt.rcParams['axes.prop_cycle'].by_key()['color']

def process_dlt(runid, num_camera, fps=60, **kwargs):
    # Output Path - Default path is the data directory
    output_points_path = PREPROCESSED_POSITION_PATH.format(runid)

    # Read Calibration Parameters
    dlt = DLT(calibration_path=CALIBRATION_PATH)
    dlt.load()

    # Read Data Points
    timelength = None
    txyz = []
    points, tags = [], []
    tags_count = defaultdict(int)
    for camid in range(1, num_camera+1):
        points_path = TRACKING_FILE.format(camid, runid)
        points_data = np.load(points_path, allow_pickle=True)
        points.append(points_data['points'])
        tags.append(points_data['tags'])
        for tag in points_data['tags']:
            tags_count[tag] += 1
        timelength = min(timelength, points[-1].shape[0]) if timelength is not None else points[-1].shape[0]
    timestamps = np.arange(0, timelength) * (1.0/fps)

    # DLT Interpolation
    result_tags = []
    result_points = []
    result_cond = []
    observing_camera_count = []
    for tag, count in tags_count.items():
        if tag in EXCLUDE_TAGS:
            continue
        if count < 2: # At least 2 camera must see the point to interpolate
            continue 
        else:
            observing_camera_count.append(count)
        result_tags.append(tag)
        point_collections_for_tag = {}
        for camid, (camera_tag, camera_points) in enumerate(zip(tags, points)):
            camid += 1
            if tag not in camera_tag:
                continue
            point_index = camera_tag.tolist().index(tag)
            point_collections_for_tag[camid] = camera_points[:timelength,point_index]
        txyz = []
        conds = []
        for time in range(timelength):
            uvs = {}
            for camid, p in point_collections_for_tag.items():
                uvs[camid] = p[time]
            _xyz, cond = dlt.map(uvs)
            txyz.append(_xyz)
            conds.append(cond)
        result_points.append(txyz)
        result_cond.append(conds)
    result_points = np.array(result_points)
    result_cond = np.array(result_cond)
    print('Process tags:')
    print('Total number of processed timesteps: {}'.format(timelength))
    print('Total number of processed points: {}'.format(result_points.shape[1]))
    print('Used Tags: ', result_tags)
    print('Final result shape: {}'.format(result_points.shape))

    # Move to simulation space
    initial_dlt_space = result_points[:,0,:]
    initial_dlt_cond = result_cond[:,0]
    initial_dlt_cond_max = initial_dlt_cond.max()
    initial_sim_space = np.empty_like(initial_dlt_space)
    for idx, tag in enumerate(result_tags):
        initial_sim_space[idx,:] = three_ring_xyz_converter(tag, kwargs.get('n_ring'), kwargs.get('ring_space'))
    reg = LinearRegression(fit_intercept=True, normalize=False).fit(initial_dlt_space, initial_sim_space)
    print('regression coefficient:\n', reg.coef_)
    print('   det=', np.linalg.det(reg.coef_))
    print('   trace=', np.trace(reg.coef_))
    print('   angle=', np.arccos((np.trace(reg.coef_)-1)/2.0))
    print('regression rank: ', reg.rank_)
    print('regression intercept: ', reg.intercept_)
    print('regression score: ', reg.score(initial_dlt_space, initial_sim_space))
    simulation_space_points = reg.predict(result_points.reshape([-1, 3]))
    simulation_space_points = simulation_space_points.reshape([-1, timelength, 3])

    #result_points = simulation_space_points

    # Save Points
    np.savez(
        output_points_path,
        time=timestamps,
        position=result_points,
        #simulation_space_points=simulation_space_points,
        tags=result_tags,
        #origin_offset=center,
        #normal_pob=np.array(plane.normal)
    )
    print('Points saved at - {}'.format(output_points_path))
    print('')

    fig = plt.figure(1, figsize=(10,8))
    ax = plt.axes(projection="3d")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    for i in range(len(result_tags)):
        #print(result_tags[i], ': ', initial_dlt_cond[i])
        ax.scatter(*initial_dlt_space[i], color=color_scheme[observing_camera_count[i]])
        #c = int(initial_dlt_cond[i]*100/initial_dlt_cond_max)-1
        #ax.scatter(*initial_dlt_space[i], c=c, cmap='viridis')
        #ax.scatter(*initial_sim_space[i])
        ax.text(*initial_dlt_space[i], result_tags[i], size=10, zorder=1, color='k')
    # draw cube
    cx = np.array([1, 15]) * 0.02
    cy = np.array([1, 12]) * 0.04
    cz = np.array([1, 10]) * 0.04
    for s, e in combinations(np.array(list(product(cx, cy, cz))), 2):
        if np.sum(np.abs(s-e)) >= 0.2:
            ax.plot3D(*zip(s, e), color="b")
    ax.view_init(elev=-90, azim=-70)

    #plt.show()

    return

if __name__=="__main__":
    for runid in range(1,2):
        process_dlt(runid=runid, num_camera=5, fps=60, n_ring=9,
                ring_space=[0.0465, 0.04125, 0.038, 0.0355, 0.0350, 0.0485, 0.030, 0.0320, 0.0345])
