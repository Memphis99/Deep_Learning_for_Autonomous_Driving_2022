from load_data import load_data
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

data_path = 'data/data.p'

data = load_data(data_path)

points = data['velodyne']

#p_coord=velodyne[:, [0,1]]
#p_intensity=velodyne[:, [3]]

x_max = max(points[:, 0])
y_max = max(points[:, 1])

x_min = min(points[:, 0])
y_min = min(points[:, 1])

image = np.zeros([int(x_max-x_min)*5+5, int(y_max-y_min)*5+5])

points[:, 0] -= x_min
points[:, 1] -= y_min

#plt.scatter(p_coord[:, 0], p_coord[:, 1])
#plt.show()

for p in points:
    x = int(np.floor(p[0]*5))
    y = int(np.floor(p[1]*5))

    if p[3] > image[x][y] and p[2]<3:
        image[x][y]=p[3]

image = ndimage.rotate(image, 90)

BEV = plt.imshow(image, cmap='gray')
plt.show()
