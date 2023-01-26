from load_data import load_data
import matplotlib.pyplot as plt
import matplotlib.image as pimg
import numpy as np
from data_utils import load_from_bin, depth_color, calib_velo2cam, calib_cam2cam, compute_timestamps, \
    load_oxts_velocity, load_oxts_angular_rate
from matplotlib import colors

data_path = 'data/problem_4/image_02/data/0000000037.png'
image = pimg.imread(data_path)

velodyne_path = 'data/problem_4/velodyne_points/data/0000000037.bin'
v_points = load_from_bin(velodyne_path)
v_points_original = load_from_bin(velodyne_path)

t_image_path = 'data/problem_4/velodyne_points/timestamps.txt'
t_image = compute_timestamps(t_image_path, '00000000037')

t_velo_path_start = 'data/problem_4/velodyne_points/timestamps_start.txt'
t_velo_path_end = 'data/problem_4/velodyne_points/timestamps_end.txt'

t_velo_start = compute_timestamps(t_velo_path_start, '00000000037')
t_velo_end = compute_timestamps(t_velo_path_end, '00000000037')

velocity_path = 'data/problem_4/oxts/data/0000000037.txt'
velocity = load_oxts_velocity(velocity_path)

angular_velocity = load_oxts_angular_rate(velocity_path)[2]

IMU_velo_path = 'data/problem_4/calib_imu_to_velo.txt'
velo_to_cam_path = 'data/problem_4/calib_velo_to_cam.txt'
cam_to_cam_path = 'data/problem_4/calib_cam_to_cam.txt'

R_velo_to_cam, T_velo_to_cam = calib_velo2cam(velo_to_cam_path)
H_cam_to_cam_prov = calib_cam2cam(cam_to_cam_path, '02')
R_IMU_velo, T_IMU_velo = calib_velo2cam(IMU_velo_path)

H_velo_to_cam_prov = np.insert(R_velo_to_cam, 3, np.transpose(T_velo_to_cam), axis=1)
H_IMU_velo_prov = np.insert(R_IMU_velo, 3, np.transpose(T_IMU_velo), axis=1)

H_IMU_velo = np.insert(H_IMU_velo_prov, 3, [0, 0, 0, 1], axis=0)
H_velo_to_cam = np.insert(H_velo_to_cam_prov, 3, [0, 0, 0, 1], axis=0)
H_cam_to_cam = np.insert(H_cam_to_cam_prov, 3, [0, 0, 0, 1], axis=0)
angle = 2 * np.pi * (t_image - t_velo_start) / (t_velo_end - t_velo_start)

if angle > np.pi:
    angle = angle - 2 * np.pi

v_points_angles = np.arctan2(v_points[:, 1], v_points[:, 0])
rotation_angles = v_points_angles
i = 0

for delta in v_points_angles:
    if delta < angle < 0:
        v_points_angles[i] += 2 * np.pi
    elif delta > angle > 0:
        v_points_angles[i] -= 2 * np.pi
    i += 1

delta_t = v_points_angles * (t_velo_end - t_velo_start) / (2 * np.pi)

v_points = np.matrix(v_points)
delta_t = np.matrix(delta_t)
delta_t = np.transpose(delta_t)
velocity = np.matrix(velocity)
velocity = np.insert(velocity, 3, 1, axis=1)
velocity = H_IMU_velo @ np.transpose(velocity)
velocity = np.transpose(velocity[0:3])

new_points = v_points - delta_t * velocity
new_points = np.array(new_points)
new_points = np.insert(new_points, 3, np.ones(len(new_points)), axis=1)
v_points_original = np.insert(v_points_original, 3, np.ones(len(v_points_original)), axis=1)

image_points = []
original_image = []
distance_vect = []
distance_vect_or = []


#rotation_angles = np.transpose(np.matrix(2 * np.pi - rotation_angles))
rotation_angles = np.transpose(np.matrix(np.arctan2(new_points[:, 1], new_points[:, 0])))
rotation_angles += delta_t * angular_velocity

rotation_angles = np.array(rotation_angles)
new_points = np.matrix(new_points)
ray = np.sqrt(np.array(np.square(new_points[:, 0]) + np.square(new_points[:, 1])))
ray = np.array(ray)

new_points = np.c_[np.array(ray * np.cos(rotation_angles[:])), np.array(ray * np.sin(rotation_angles[:])), new_points[:, 2], np.ones((len(new_points), 1))]
new_points = np.array(new_points)
#new_points += rotation


for p in new_points:
    if p[0] > 0:  # if x(i) is > 0 (values which are not behind the velodyne)
        point1 = H_cam_to_cam @ H_velo_to_cam @ p
        point2 = point1[0:2] / point1[2]
        distance = np.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2)
        distance_vect.append(distance)
        image_points.append(point2)

for q in v_points_original:
    if q[0] > 0:  # if x(i) is > 0 (values which are not behind the velodyne)
        point3 = H_cam_to_cam @ H_velo_to_cam @ q
        point4 = point3[0:2] / point3[2]
        distance1 = np.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2)
        distance_vect_or.append(distance1)
        original_image.append(point4)

original_image = np.array(original_image)
image_points = np.array(image_points)
distance_vect = np.array(distance_vect)
distance_vect_or = np.array(distance_vect_or)

colours_corrected_HSV = depth_color(distance_vect)
colours_original_HSV = depth_color(distance_vect_or)

colours_corrected_HSV_normalized = [colours_corrected_HSV / 135, np.ones(len(colours_corrected_HSV)),
                                    np.ones(len(colours_corrected_HSV))]
colours_corrected_RGBA = colors.hsv_to_rgb(np.transpose(colours_corrected_HSV_normalized))
colours_original_HSV_normalized = [colours_original_HSV / 135, np.ones(len(colours_original_HSV)),
                                   np.ones(len(colours_original_HSV))]
colours_original_RGBA = colors.hsv_to_rgb(np.transpose(colours_original_HSV_normalized))

plt.figure(1)
plt.scatter(image_points[:, 0], image_points[:, 1], s=1, edgecolors=colours_corrected_RGBA, marker='.')
plt.imshow(image)
plt.show()

plt.figure(2)
plt.scatter(original_image[:, 0], original_image[:, 1], s=1, edgecolors=colours_original_RGBA, marker='.')
plt.imshow(image)
plt.show()

