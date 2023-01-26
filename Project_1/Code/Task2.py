from load_data import load_data
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

data_path = 'data/data.p'

data = load_data(data_path)

points = data['velodyne']

image = data['image_2']

K = data['K_cam2']

T_cam2_velo = data['T_cam2_velo']
T_cam0_velo = data['T_cam0_velo']

P = data['P_rect_20']

markers = []
colours = []

for i in range(len(points)):
    if points[i][0] > 0:   #if x(i) is > 0 (values which are not behind the velodyne)
        point = np.transpose(np.append(points[i][0:3], 1))
        point1 = T_cam2_velo @ point
        point2 = K @ point1[0:3]
        point3 = point2[0:2] / point2[2]
        markers.append(point3)

        label = int(data['sem_label'][i])
        colour = data['color_map'][label]
        colour = [colour[2], colour[1], colour[0]]

        colours.append(colour)

markers = np.array(markers)
colours = np.array(colours) / 255

plt.scatter(markers[:, 0], markers[:, 1], s=0.07, c=colours, marker='.')

objects = data['objects']
bb = []
i = -1

for car in objects:
    i += 1

    h_car = car[8]
    l_car = car[9]
    w_car = car[10]
    y_angle = car[14]

    bbpoints = [[-w_car/2, 0, l_car/2], [w_car/2, 0, l_car/2], [w_car/2, 0, -l_car/2], [-w_car/2, 0, -l_car/2], \
    [-w_car/2, -h_car, l_car/2], [w_car/2, -h_car, l_car/2], [w_car/2, -h_car, -l_car/2], [-w_car/2, -h_car, -l_car/2]]

    R = np.array([[np.cos(y_angle), 0, np.sin(y_angle)], [0, 1, 0], [-np.sin(y_angle), 0, np.cos(y_angle)]])
    r_0c = np.array([car[11], car[12], car[13]]) #we have roc from the data
    Tprov = np.insert(R, 3, r_0c, axis=1)
    T_cam0_car = np.insert(Tprov, 3, [0, 0, 0, 1], axis=0) #homogeneous transformation from cam0 frame to car frame

    T_cam2_cam0 = T_cam2_velo @ np.linalg.inv(T_cam0_velo) #multiply the inverse of inv(T_cam0_velo)*T_cam2_velo
    T_cam2_car = T_cam2_cam0 @ T_cam0_car #T2c = T20*T0c

    for b in bbpoints:
        bbpixels = np.append(b, 1)
        bbpixels1 = T_cam2_car @ bbpixels
        bbpixels2 = K @ bbpixels1[0:3]
        bbpixels3 = bbpixels2[0:2] / bbpixels2[2]

        bb.append(bbpixels3)

    plt.gca().add_patch(Polygon(bb[i*8:i*8+4], closed=True, fill=False, ec='lawngreen', lw=1.8))
    plt.gca().add_patch(Polygon(bb[i*8+4:i*8+8], closed=True, fill=False, ec='lawngreen', lw=1.8))

    plt.gca().add_patch(Polygon([bb[i*8], bb[i*8+4], bb[i*8+7], bb[i*8+3]], closed=True, fill=False, ec='lawngreen', lw=1.8))
    plt.gca().add_patch(Polygon([bb[i*8+1], bb[i*8+5], bb[i*8+6], bb[i*8+2]], closed=True, fill=False, ec='lawngreen', lw=1.8))

bb = np.array(bb)

plt.imshow(image)
plt.show()


