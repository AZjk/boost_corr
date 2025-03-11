import os
import time
import sys
from PyQt5 import uic, Qt
from PyQt5.QtWidgets import (QFileDialog, QMainWindow, QApplication, QWidget,
                             QVBoxLayout, QHBoxLayout, QSplitter)
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from gpu_corr_server import GPUServer, get_system_information
import json
import torch
from pyqtgraph import DataTreeWidget, ImageView
import traceback
import pyqtgraph as pg
import numpy as np

pg.setConfigOption('imageAxisOrder', 'row-major')

home_dir = os.path.join(os.path.expanduser('~'), '.xpcs_boost_server')
if not os.path.isdir(home_dir):
    os.mkdir(home_dir)


class SetupPreviewer(QSplitter):
    def __init__(self):
        super(SetupPreviewer, self).__init__()
        self.initUI()

    def initUI(self):
        hbox12 = QHBoxLayout()
        self.img1 = ImageView()
        self.img2 = ImageView()
        hbox12.addWidget(self.img1)
        hbox12.addWidget(self.img2)

        hbox34 = QHBoxLayout()
        self.img3 = ImageView()
        self.img4 = ImageView()
        hbox34.addWidget(self.img3)
        hbox34.addWidget(self.img4)
        vbox_img = QVBoxLayout()
        vbox_img.addLayout(hbox12)
        vbox_img.addLayout(hbox34)
        img_widget = QWidget()
        img_widget.setLayout(vbox_img)

        # hbox = QSplitter()
        self.data_tree = DataTreeWidget()
        self.data_tree.setMaximumWidth(400)
        self.addWidget(self.data_tree)
        self.addWidget(img_widget)

        self.setGeometry(300, 300, 1200, 800)
        self.setWindowTitle('Setup Previewer')
        self.show()

    def set_data(self, fname, setup):
        img_hdl = (self.img1, self.img2, self.img3, self.img4)
        for hdl, key in zip(img_hdl, ("qMap", "qCMap", "qRMap", "mask")):
            hdl.setImage(setup[key])
            text1 = pg.TextItem(text=f'{key}', color=(0, 0, 255))
            plot_view = hdl.getView()
            plot_view.addItem(text1)
            setup.pop(key, None)
        self.data_tree.setData(setup)
        self.setWindowTitle(f'Setup Previewer: {fname}')


class Ui(QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('gpuBoostServer.ui', self)
        self.btn_start.clicked.connect(self.start_server)
        self.btn_stop.clicked.connect(self.stop_server)
        self.home_dir = home_dir

        self.is_running = False
        self.server_kwargs = None
        self.server = None
        self.update_time = 0.500  # 500 ms

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_status)

        self.btn_select_result_dir.clicked.connect(self.select_result_dir)
        self.btn_submit.clicked.connect(self.submit_jobs)
        # plot the load
        self.load_window = 256

        xline = np.arange(self.load_window) * self.update_time
        self.load_xline = xline - np.max(xline)
        self.load_plot.setBackground('w')
        self.load_plot.setLabel('left', 'jobs/s')
        self.load_plot.setLabel('bottom', 'Time', units='s')
        self.load_plot.hideAxis('bottom')
        self.load_plot.setMouseEnabled(x=False, y=False)

        self.load_save_setting(mode="load")
        sys_info = get_system_information()
        self.model = QStandardItemModel()
        for n in range(sys_info['num_gpus']):
            name = sys_info['device'][n]['name']
            label = name + f'_{n}'
            item = QStandardItem(label)
            item.setCheckState(True)
            item.setCheckable(True)
            self.model.appendRow(item)

        self.device_list.setModel(self.model)
        self.show()

    def select_setup_file(self, idx=1):
        if self.is_running:
            self.statusbar.showMessage('Server is running now.', 2000)
            return
        assert idx in (1, 2, 3)
        display = f'setup_fname_{idx}'
        f = QFileDialog.getOpenFileName(self, f'select setup file {idx}')[0]
        if os.path.isfile(f):
            self.__dict__[display].setText(f)
        else:
            self.statusbar.showMessage('the setup file is not valid.')
        return

    def select_result_dir(self):
        if self.is_running:
            self.statusbar.showMessage('Server is running now.', 2000)
            return

        d = QFileDialog.getExistingDirectory(self, f'select result directory')
        if os.path.isdir(d):
            self.result_dir.setText(d)
        else:
            self.statusbar.showMessage('the result directory is not valid.')

    def submit_jobs(self):
        if not self.is_running:
            self.statusbar.showMessage('Server is not running. quit', 2000)
            return

        flists = QFileDialog.getOpenFileName(self,
                                             f'select raw files to reduce',
                                             directory='/clhome/MQICHU',
                                             filter="TXT (*.txt)")
        # filter="TIFF (*.tif);;HDF (*.h5 *.hdf)")

        # self.server.submit_jobs(flists[0])
        # self.server.submit_(flists[0])
        print(flists[0])
        self.server.process_all_files(flists[0])

    def update_status(self):
        if self.server is not None:
            flag, message, jobs_info = self.server.get_status(self.load_window)
        else:
            flag, message, jobs_info = False, "stopped", None

        if flag:
            self.lineEdit_8.setStyleSheet("background-color: rgb(0, 255, 0)")
            yline = jobs_info[:, 3] / self.update_time
            self.load_plot.plot(self.load_xline,
                                yline,
                                fillLevel=0,
                                clear=True,
                                brush=(50, 50, 200, 200))
        else:
            self.lineEdit_8.setStyleSheet("background-color: rgb(255, 255, 0)")
        message = str(time.asctime()) + ' | ' + message
        self.lineEdit_8.setText(message)

    def start_server(self):
        if self.is_running:
            self.statusbar.showMessage('Server is running.', 1000)
            return

        self.statusbar.showMessage('Server is starting', 8000)

        kwargs = {}
        kwargs['port'] = self.port.value()
        kwargs['output'] = self.result_dir.text()
        kwargs['worker_per_gpu'] = 1
        self.server_kwargs = kwargs

        gpu_selection = []
        for n in range(self.model.rowCount()):
            idx = self.model.index(n, 0)
            flag = self.model.itemFromIndex(idx).checkState()
            gpu_selection.append(flag)

        if sum(gpu_selection) == 0:
            self.statusbar.showMessage('None of the GPUs is selected', 1000)
            return
        else:
            kwargs['gpu_selection'] = gpu_selection

        try:
            self.server = GPUServer(**kwargs)
        except Exception as e:
            print('failed to start server')
            traceback.print_exc()

        time.sleep(1)
        self.timer.start(int(self.update_time * 1000))
        self.is_running = True
        self.lineEdit.setText(self.server.ip_addr)
        self.statusbar.showMessage('Server started', 1000)

    def enable_disable_panel(self, flag=True, idx=0):
        pass
        # self.gb_det_1.setEnabled(flag)

    def stop_server(self):
        self.statusbar.showMessage('Server is stopping', 8000)
        if self.is_running:
            self.timer.stop()
            self.server.stop_server()
            self.server = None
            self.is_running = False
        self.update_status()
        self.statusbar.showMessage('Server stopped', 1000)

    def load_save_setting(self, mode='save'):
        text_field = ["watch_pv", "watch_folder", "result_dir"]

        num_field = ["port"]
        assert mode in ("save", "load"), "mode not supported."

        fname = os.path.join(self.home_dir, 'last_setting.json')
        if mode == "save":
            kwargs = {}
            for key in text_field:
                kwargs[key] = self.__dict__[key].text()
            for key in num_field:
                kwargs[key] = self.__dict__[key].value()
            with open(fname, 'w') as f:
                json.dump(kwargs, f, indent=4)

        elif mode == "load":
            try:
                with open(fname, 'r') as f:
                    kwargs = json.load(f)
                for key in text_field:
                    self.__dict__[key].setText(kwargs[key])
                for key in num_field:
                    self.__dict__[key].setValue(kwargs[key])
            except Exception:
                pass

    def closeEvent(self, event):
        self.stop_server()
        self.load_save_setting(mode="save")

        super(QMainWindow, self).closeEvent(event)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    app = QApplication(sys.argv)
    window = Ui()
    app.exec_()
