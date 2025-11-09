#!/usr/bin/env python3

import os
import re
import sys

import laspy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector
import numpy as np
from PyQt5 import QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget,
  QLabel, QPushButton, QFileDialog, QSlider, QDialog,QLineEdit, QDialogButtonBox,
  QShortcut, QHBoxLayout, QSpacerItem, QSizePolicy)


class SliceInputDialog(QDialog):
    def __init__(self, parent=None):
        super(SliceInputDialog, self).__init__(parent)

        self.setWindowTitle("Slice Configuration")

        self.thickness_label = QLabel("Slice Thickness:")
        self.thickness_input = QLineEdit(self)
        self.thickness_input.setText("0.02")
        self.thickness_input.setValidator(QtGui.QDoubleValidator())

        self.interval_label = QLabel("Slice Interval:")
        self.interval_input = QLineEdit(self)
        self.interval_input.setText("1.0")
        self.interval_input.setValidator(QtGui.QDoubleValidator())

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addWidget(self.thickness_label)
        layout.addWidget(self.thickness_input)
        layout.addWidget(self.interval_label)
        layout.addWidget(self.interval_input)
        layout.addWidget(self.button_box)

        self.setLayout(layout)

    def get_values(self):
        thickness = float(self.thickness_input.text())
        interval = float(self.interval_input.text())
        return thickness, interval


class PointCloudViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.slice_thickness = 0.1
        self.slice_spacing = 5.0
        self.current_slice_index = 0
        self.slices = []
        self.selections = []
        self.results = []
        self.slice_heights = []
        
        self.global_min = None
        self.global_max = None
        self.last_filename = None

        self.initUI()

    def initUI(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        self.canvas = FigureCanvas(Figure())
        layout.addWidget(self.canvas)

        self.ax = self.canvas.figure.add_subplot(111)
        self.ax.set_aspect('equal', adjustable='datalim')

        self.selector = LassoSelector(self.ax, onselect=self.on_points_selected)

        controls_layout = QHBoxLayout()
        
        # Load and Save buttons
        btn_load = QPushButton('Load LAS (L)', self)
        btn_load.clicked.connect(self.load_las_file)
        controls_layout.addWidget(btn_load)

        btn_save_results = QPushButton('Save (S)', self)
        btn_save_results.clicked.connect(self.save_results)
        controls_layout.addWidget(btn_save_results)

        # Previous and Next buttons
        btn_prev_slice = QPushButton('Previous (X) ', self)
        btn_prev_slice.clicked.connect(self.prev_slice)
        controls_layout.addWidget(btn_prev_slice)

        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(100)  # Set the maximum according to your needs
        self.slice_slider.setSingleStep(1)
        self.slice_slider.valueChanged.connect(self.on_slice_changed)
        controls_layout.addWidget(self.slice_slider)

        btn_next_slice = QPushButton('Next (V)', self)
        btn_next_slice.clicked.connect(self.next_slice)
        controls_layout.addWidget(btn_next_slice)
        
        btn_fit_circle = QPushButton('Fit Circle (C)', self)
        btn_fit_circle.clicked.connect(self.fit_circle)
        controls_layout.addWidget(btn_fit_circle)
        
        btn_try_load_next = QPushButton('Try Load Next (N)', self)
        btn_try_load_next.clicked.connect(self.try_load_next)
        controls_layout.addWidget(btn_try_load_next)

        # Spacer to push the buttons to the sides
        controls_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        layout.addLayout(controls_layout)

        # Add shortcuts
        self.shortcut_calc_circle = QShortcut(QtGui.QKeySequence("c"), self)
        self.shortcut_calc_circle.activated.connect(self.fit_circle)

        self.shortcut_prev_slice = QShortcut(QtGui.QKeySequence("x"), self)
        self.shortcut_prev_slice.activated.connect(self.prev_slice)

        self.shortcut_next_slice = QShortcut(QtGui.QKeySequence("v"), self)
        self.shortcut_next_slice.activated.connect(self.next_slice)

        self.shortcut_load = QShortcut(QtGui.QKeySequence("l"), self)
        self.shortcut_load.activated.connect(self.load_las_file)

        self.shortcut_save = QShortcut(QtGui.QKeySequence("s"), self)
        self.shortcut_save.activated.connect(self.save_results)

        self.shortcut_try_load_next = QShortcut(QtGui.QKeySequence("n"), self)
        self.shortcut_try_load_next.activated.connect(self.try_load_next)

        self.show()

    def load_las_file(self):
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("LAS files (*.las)")
        file_dialog.setFileMode(QFileDialog.ExistingFile)

        if file_dialog.exec_():
            filename = file_dialog.selectedFiles()[0]

            input_dialog = SliceInputDialog(self)
            if input_dialog.exec_():
                thickness, interval = input_dialog.get_values()
                self.load_point_cloud(filename, thickness, interval)

    def try_load_next(self):
        if self.last_filename is None:
            return
        
        new_filename = re.sub(r"(\d{3})", lambda match: f"{int(match.group(1)) + 1:03d}", self.last_filename)

        input_dialog = SliceInputDialog(self)
        if input_dialog.exec_():
            thickness, interval = input_dialog.get_values()
            self.load_point_cloud(new_filename, thickness, interval)

    def load_point_cloud(self, filename, thickness, interval):
        if self.slices != []:
            self.slices = []
            self.selections = []
            self.results = []
            self.slice_heights = []
            self.slice_slider.setMaximum(0)
            self.current_slice_index = 0
        if not os.path.exists(filename):
            return
        las_file = laspy.read(filename)
        self.last_filename = filename
        points = np.vstack([las_file.x, las_file.y, las_file.z]).T
        self.global_min = np.min(points[:, :2], axis=0)
        self.global_max = np.max(points[:, :2], axis=0)
        
        self.slices = []
        self.slice_heights = np.concatenate(
            (np.arange(points[:, 2].min(), points[:, 2].min() + 3, interval / 2),
            np.arange(points[:, 2].min() + 3, points[:, 2].max(), interval))
        )
        for slice_height in self.slice_heights:
            sliced_pc = points[
                np.logical_and(points[:, 2] >= slice_height - thickness / 2, points[:, 2] < slice_height + thickness / 2)
            ]
            if len(sliced_pc) > 0:
                self.slices.append(sliced_pc)
        self.slice_slider.setMaximum(len(self.slices) - 1)
        self.slice_slider.setValue(0)
        self.selections = [set() for i in range(len(self.slices))]
        self.results = [{} for i in range(len(self.slices))]
        
        self.slice_thickness = thickness
        self.slice_interval = interval

        self.update_plot()

    def update_plot(self):
        if self.slices == []:
            return
        self.ax.clear()
        points = self.slices[self.current_slice_index]
        self.ax.scatter(points[:, 0], points[:, 1], s=1, c='b')
        self.ax.set_title(f"Slice {self.current_slice_index} at height {self.slice_heights[self.current_slice_index]:.2f} m")
        self.ax.set_xlim(self.global_min[0]-1, self.global_max[0]+1)
        self.ax.set_ylim(self.global_min[1]-1, self.global_max[1]+1)
        
        cur_selected_points = points[list(self.selections[self.current_slice_index])]
        if cur_selected_points.shape[0] > 0:
            selected = np.array(list(cur_selected_points))
            self.ax.scatter(selected[:, 0], selected[:, 1], s=2, c='g')
        
        # check if result is not empty
        result = self.results[self.current_slice_index]
        if result != {}:
            c_x = result['center_x']
            c_y = result['center_y']
            r = result['radius']
            self.ax.add_patch(Circle((c_x, c_y), r, color='r', fill=False, lw=2))
        
        self.canvas.draw()

    def on_points_selected(self, verts):
        if self.slices == []:
            return
        path = Path(verts)
        cur_points = self.slices[self.current_slice_index]
        
        selected_indices = np.nonzero(path.contains_points(cur_points[:, :2]))[0]
        if QApplication.keyboardModifiers() == Qt.ControlModifier:
            self.selections[self.current_slice_index] -= set(selected_indices)
        else:
            self.selections[self.current_slice_index] |= set(selected_indices)
        self.selector.clear()
            
        self.update_plot()

    def fit_circle_bullock(self, points):
        if self.slices == []:
            return
        # normalize points
        points_mean = np.mean(points, axis=0)
        u = points[:, 0] - points_mean[0]
        v = points[:, 1] - points_mean[1]
        # pre-calculate summands
        S_uu = np.sum(np.power(u, 2))
        S_vv = np.sum(np.power(v, 2))
        S_uv = np.sum(u * v)
        S_uuu_uvv = np.sum(np.power(u, 3) + u * np.power(v, 2))
        S_vvv_vuu = np.sum(np.power(v, 3) + v * np.power(u, 2))
        # calculate circle center in normalized coordinates and radius
        v_c = (S_uuu_uvv / (2 * S_uu) - S_vvv_vuu / (2 * S_uv)) / (
            S_uv / S_uu - S_vv / S_uv + 1e-12
        )
        u_c = (S_uuu_uvv / (2 * S_uv) - S_vvv_vuu / (2 * S_vv)) / (
            S_uu / S_uv - S_uv / S_vv + 1e-12
        )
        r = np.sqrt(np.power(u_c, 2) + np.power(v_c, 2) + (S_uu + S_vv) / points.shape[0])
        # denormalize
        x_c, y_c = points_mean[0] + u_c, points_mean[1] + v_c
        return x_c, y_c, r

    def fit_circle(self):
        if not self.selections:
            return

        cur_points = self.slices[self.current_slice_index]
        cur_selection = list(self.selections[self.current_slice_index])
        selected_points = cur_points[cur_selection]
        c_x, c_y, r = self.fit_circle_bullock(selected_points)

        self.results[self.current_slice_index] = {
            'slice_height': self.slice_heights[self.current_slice_index],
            'num_points': len(self.selections),
            'center_x': c_x,
            'center_y': c_y,
            'radius': r
        }

        self.update_plot()

    def save_results(self):
        if not self.results:
            return

        if self.last_filename:
            filename_proposal = self.last_filename.replace('.las', '_results.csv')
        else:
            filename_proposal = ""
        filename, _ = QFileDialog.getSaveFileName(self, 'Save Results', filename_proposal, 'CSV files (*.csv)')

        if filename:
            with open(filename, 'w') as f:
                f.write("slice_height,num_points,center_x,center_y,radius\n")
                for result in self.results:
                    if result != {}:
                        f.write(f"{result['slice_height']},{result['num_points']},{result['center_x']},{result['center_y']},{result['radius']}\n")

    def on_slice_changed(self, value):
        if self.slices == []:
            return
        self.current_slice_index = value
        self.update_plot()

    def prev_slice(self):
        if self.slices == []:
            return
        if self.current_slice_index > 0:
            self.current_slice_index -= 1
            self.slice_slider.setValue(self.current_slice_index)
            self.update_plot()

    def next_slice(self):
        if self.slices == []:
            return
        if self.current_slice_index < self.slice_slider.maximum():
            self.current_slice_index += 1
            self.slice_slider.setValue(self.current_slice_index)
            self.update_plot()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PointCloudViewer()
    sys.exit(app.exec_())