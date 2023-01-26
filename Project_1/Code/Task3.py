from load_data import load_data
import matplotlib.pyplot as plt
import numpy as np
from data_utils import line_color
from matplotlib import colors

data_path = 'data/data.p'
data = load_data(data_path)

points = data['velodyne']
image = data['image_2']

pixel_y_max = np.size(image, 0)
pixel_x_max = np.size(image, 1)

T_cam2_velo = data['T_cam2_velo']
K = data['K_cam2']

good_points = []
colours = []
index = []

angles = np.arctan2(points[:, 2], np.sqrt(points[:, 0]**2 + points[:, 1]**2))

max_angle = max(angles) * 180/np.pi
min_angle = min(angles) * 180/np.pi
res = round(((max_angle - min_angle)/64), 2)

for j in range(len(points)):
    p = points[j]
    if p[0] > 0:   #if x(i) is > 0 (values which are not behind the velodyne)
        point = np.transpose(np.append(p[0:3], 1))
        point1 = T_cam2_velo @ point
        point2 = K @ point1[0:3]
        point3 = point2[0:2] / point2[2]
        if point3[0] >= 0 and point3[0] <= pixel_x_max and point3[1] >= 0 and point3[1] <= pixel_y_max:
            k = 0
            for i in np.arange(min_angle, max_angle, res):
                k += 1
                if i * (np.pi / 180) <= angles[j] <= (i+res) * (np.pi / 180):
                    good_points.append(point3)
                    index.append(k)

good_points = np.array(good_points)
index = np.array(index)

print(index)

colour_HSV = line_color(index)
colour_HSV_normalized = [colour_HSV/135, np.ones(len(colour_HSV)), np.ones(len(colour_HSV))]
colour_RGBA = colors.hsv_to_rgb(np.transpose(colour_HSV_normalized))

plt.scatter(good_points[:, 0], good_points[:, 1], s=1, edgecolors=colour_RGBA, marker='.')
plt.imshow(image)
plt.show()
