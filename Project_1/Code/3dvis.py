# Deep Learning for Autonomous Driving
# Material for Problem 2 of Project 1
# For further questions contact Ozan Unal, ozan.unal@vision.ee.ethz.ch

import vispy
from vispy.scene import visuals, SceneCanvas
import numpy as np
import os
from load_data import load_data


class Visualizer():
    def __init__(self):
        self.canvas = SceneCanvas(keys='interactive', show=True)
        self.grid = self.canvas.central_widget.add_grid()
        self.view = vispy.scene.widgets.ViewBox(border_color='white',
                                                parent=self.canvas.scene)
        self.grid.add_widget(self.view, 0, 0)

        # Point Cloud Visualizer
        self.sem_vis = visuals.Markers()
        self.view.camera = vispy.scene.cameras.TurntableCamera(up='z', azimuth=90)
        self.view.add(self.sem_vis)
        visuals.XYZAxis(parent=self.view.scene)

        # Object Detection Visualizer
        self.obj_vis = visuals.Line()
        self.view.add(self.obj_vis)
        self.connect = np.asarray([[6, 5], [6, 7], [6, 2],
                                   [4, 5], [4, 7], [4, 0],
                                   [1, 5], [1, 2], [1, 0],
                                   [3, 7], [3, 2], [3, 0]])

    def update(self, points, sem_label, colours):
        '''
        :param points: point cloud data
                        shape (N, 3)
        Task 2: Change this function such that each point
        is colored depending on its semantic label
        '''
        colour_vec = np.zeros([len(points), 3])
        for i in range(len(points)):
            label = sem_label[i]
            colour = colours[int(label)]
            colour = [colour[2], colour[1], colour[0]]
            colour_vec[i][:] = np.array(colour) / 255

        self.sem_vis.set_data(points, size=3, face_color=colour_vec)

    def update_boxes(self, corners):
        '''
        :param corners: corners of the bounding boxes
                        shape (N, 8, 3) for N boxes
        (8, 3) array of vertices for the 3D box in
        following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
        If you plan to use a different order, you can
        change self.connect accordinly.
        '''
        for i in range(corners.shape[0]):
            connect = np.concatenate((connect, self.connect + 8 * i), axis=0) \
                if i > 0 else self.connect
        self.obj_vis.set_data(corners.reshape(-1, 3),
                              connect=connect,
                              width=2,
                              color=[0, 1, 0, 1])


if __name__ == '__main__':
    data = load_data('data/data.p')  # Change to data.p for your final submission
    visualizer = Visualizer()
    label = data['sem_label']
    colour = data['color_map']
    objects = data['objects']
    T_cam2_velo = data['T_cam2_velo']
    T_cam0_velo = data['T_cam0_velo']

    visualizer.update(data['velodyne'][:, :3], label, colour)

    bb = np.zeros((len(objects), 8, 3))
    i = -1
    for car in objects:
        i += 1
        h_car = car[8]
        l_car = car[9]
        w_car = car[10]
        y_angle = car[14]

        bbpoints = [[-w_car / 2, 0, l_car / 2], [w_car / 2, 0, l_car / 2], [w_car / 2, 0, -l_car / 2],
                    [-w_car / 2, 0, -l_car / 2], \
                    [-w_car / 2, -h_car, l_car / 2], [w_car / 2, -h_car, l_car / 2], [w_car / 2, -h_car, -l_car / 2],
                    [-w_car / 2, -h_car, -l_car / 2]]

        R = np.array([[np.cos(y_angle), 0, np.sin(y_angle)], [0, 1, 0], [-np.sin(y_angle), 0, np.cos(y_angle)]])
        r_0c = np.array([car[11], car[12], car[13]])  # we have roc from the data
        Tprov = np.insert(R, 3, r_0c, axis=1)
        T_cam0_car = np.insert(Tprov, 3, [0, 0, 0, 1],
                               axis=0)  # homogeneous transformation from cam0 frame to car frame
        T_velo_car = np.linalg.inv(T_cam0_velo) @ T_cam0_car  # homogeneous transformation from car to velodyne frame

        bbpoints1 = np.insert(bbpoints, 3, 1, axis=1)
        bbpoints2 = T_velo_car @ np.transpose(bbpoints1)

        bbpoints3 = np.transpose(bbpoints2[0:3, :])
        bb[i][:][:] = bbpoints3

    visualizer.update_boxes(bb)

    '''
    Task 2: Compute all bounding box corners from given
    annotations. You can visualize the bounding boxes using
    visualizer.update_boxes(corners)
    '''
    vispy.app.run()






