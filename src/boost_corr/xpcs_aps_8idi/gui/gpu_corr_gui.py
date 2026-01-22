import os
import sys
import json
import torch
import psutil
from PyQt5 import uic, QtCore
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QListView,
                             QAbstractItemView, QTreeView)
from gpu_corr_solver import GPUSolverWorker, get_raw_meta


def get_system_information():
    sys_info = {}
    sys_info['num_gpus'] = torch.cuda.device_count()
    sys_info['num_cpus'] = psutil.cpu_count(logical=False)
    scale = 1024**3

    device = {}
    for n in range(sys_info['num_gpus']):
        a = torch.cuda.get_device_properties('cuda:%d' % n)
        device[n] = {'name': a.name, 'total_memory': a.total_memory / scale}

    cpu_ram = psutil.virtual_memory().total / scale
    # set an upper limit to cpu_ram;
    cpu_ram = min(36.0, cpu_ram)
    device[-1] = {'name': 'cpu', 'total_memory': cpu_ram}
    sys_info['device'] = device

    return sys_info


class TableDataModel(QtCore.QAbstractTableModel):
    def __init__(self, input_list=None, max_display=16384) -> None:
        super().__init__()
        if input_list is None:
            self.input_list = []
        else:
            self.input_list = input_list
        self.max_display = max_display
        self.xlabels = [
            'id', 'priority', 'gpu_id', 'status', 'frames', 'progress(%)',
            'time/freq', 'fname'
        ]

    # overwrite parent method
    def data(self, index, role):
        if role == QtCore.Qt.DisplayRole:
            x = self.input_list[len(self.input_list) - 1 - index.row()]
            ret = x.get_data()
            return ret[index.column()]

    # overwrite parent method
    def rowCount(self, index):
        return min(self.max_display, len(self.input_list))

    # overwrite parent method
    def columnCount(self, index):
        return len(self.xlabels)

    def headerData(self, section, orientation, role):
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                return self.xlabels[section]

    def extend(self, new_input_list):
        self.input_list.extend(new_input_list)
        self.layoutChanged.emit()

    def append(self, new_item):
        self.input_list.append(new_item)
        self.layoutChanged.emit()

    def replace(self, new_input_list):
        self.input_list.clear()
        self.extend(new_input_list)

    def pop(self, index):
        if 0 <= index < self.__len__():
            self.input_list.pop(index)
            self.layoutChanged.emit()

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, i):
        return self.input_list[i]

    def copy(self):
        return self.input_list.copy()

    def remove(self, x):
        self.input_list.remove(x)

    def clear(self):
        self.input_list.clear()

    def emit(self):
        self.layoutChanged.emit()


class Ui(QMainWindow):
    def __init__(self, config_fname=None):
        super(Ui, self).__init__()
        uic.loadUi('gpu_corr.ui', self)
        self.sys_info = get_system_information()
        self.gpu_id.setMaximum(self.sys_info['num_gpus'] - 1)
        self.num_worker.setMaximum(self.sys_info['num_cpus'])
        self.num_worker.setValue(self.sys_info['num_cpus'] // 2)

        self.config_fname = config_fname
        self.config = None

        # only allow one worker because the computing is intense; jobs will be
        # queued
        self.thread_pool = QtCore.QThreadPool()
        self.thread_pool.setMaxThreadCount(1)
        self.workers = TableDataModel()
        self.job_table.setModel(self.workers)

        self.job_count = 0
        self.last_raw_dir = None
        self.load_last_config()
        self.show()
        self.adjust_gpu_id()

    def load_qmap(self):
        work_dir = os.path.dirname(self.qmap.text())
        if not os.path.isdir(work_dir):
            work_dir = None
        f = QFileDialog.getOpenFileName(self, 'select qmap file',
                directory=work_dir)[0]
        if f in [None, '']:
            return
        self.config['qmap'] = f
        self.qmap.setText(f)

    def load_raw(self):
        # https://stackoverflow.com/questions/38252419
        file_dialog = QFileDialog(self, directory=self.last_raw_dir)
        file_dialog.setFileMode(QFileDialog.DirectoryOnly)
        file_dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        file_view = file_dialog.findChild(QListView, 'listView')

        # to make it possible to select multiple directories:
        if file_view:
            file_view.setSelectionMode(QAbstractItemView.MultiSelection)
        f_tree_view = file_dialog.findChild(QTreeView)
        if f_tree_view:
            f_tree_view.setSelectionMode(QAbstractItemView.MultiSelection)

        paths = None
        if file_dialog.exec():
            paths = file_dialog.selectedFiles()

        if paths in [None, '', []]:
            return

        valid_path = []
        for x in paths:
            raw, meta = get_raw_meta(x)
            if raw is not None and meta is not None:
                valid_path.append(x)

        if len(valid_path) > 0:
            self.config['raw_folder'].extend(valid_path)
            # self.raw_folder.clear()
            self.raw_folder.addItems(valid_path)
            self.last_raw_dir = os.path.dirname(valid_path[-1])

    def set_output(self):
        work_dir = os.path.dirname(self.output.text())
        if not os.path.isdir(work_dir):
            work_dir = None

        f = QFileDialog.getExistingDirectory(self, 'select saving folder',
                directory=work_dir)
        if f in [None, '']:
            return
        self.config['output'] = f
        self.output.setText(f)

    def adjust_gpu_id(self):
        gpu_id = self.gpu_id.value()
        device = self.sys_info['device'][gpu_id]
        self.gpu_id_label.setText('Processing Unit ID (%s)' % device['name'])
        self.max_memory.setMaximum(device['total_memory'] * 0.8)
        self.max_memory.setValue(device['total_memory'] * 0.5)
        return

    def update_table(self):
        self.workers.emit()

    def clear_raw_folders(self):
        self.raw_folder.clear()
        self.config['raw_folder'] = []

    def submit_job(self):
        kwargs = {}
        kwargs.update(self.config)
        raw_folders = kwargs.pop('raw_folder')
        self.raw_folder.clear()
        self.config['raw_folder'] = []

        kwargs['gpu_id'] = self.gpu_id.value()
        kwargs['batch_size'] = int(self.batch_size.currentText())
        kwargs['crop_ratio_threshold'] = self.masked_ratio_threshold.value()
        kwargs['priority'] = self.job_priority.value()
        kwargs['num_worker'] = self.num_worker.value()
        kwargs['max_memory'] = self.max_memory.value()
        if not os.path.isdir(kwargs['output']) or \
           not os.path.isfile(kwargs['qmap']):
            return

        for n in range(len(raw_folders)):
            kw = kwargs.copy()
            kw['raw_folder'] = raw_folders[n]
            self.job_count += 1
            # self.job_producer.submit_job(kwargs)
            worker = GPUSolverWorker(jid=self.job_count, **kw)
            self.workers.append(worker)
            worker.signals.status.connect(self.update_table)
            worker.signals.progress.connect(self.update_table)
            # the thread count is one; so if previous workers are running,
            # the new # woker is queued until the previous one is done;
            self.thread_pool.start(worker, priority=kw['priority'])

    def save_config(self):
        with open(self.config_fname, 'w') as fhdl:
            json.dump(self.config, fhdl)

    def load_last_config(self, ):
        if os.path.isfile(self.config_fname):
            with open(self.config_fname, 'r') as fhdl:
                self.config = json.load(fhdl)
                for key, val in self.config.items():
                    if val is None or key not in [
                            'qmap', 'output', 'raw_folder']:
                        continue
                    if key != 'raw_folder':
                        self.__dict__[key].setText(self.config[key])
                    else:
                        self.__dict__[key].addItems(self.config[key])
                        if len(self.config[key]) > 0:
                            self.last_raw_dir = os.path.dirname(self.config[key][0])
        else:
            self.config = {'qmap': None, 'raw_folder': [], 'output': None}
        return

    def closeEvent(self, e) -> None:
        self.save_config()

    def __del__(self):
        pass


home_dir = os.path.join(os.path.expanduser('~'), '.xpcs_boost')
if not os.path.isdir(home_dir):
    os.mkdir(home_dir)
config_fname = os.path.join(home_dir, 'last_config.json')
app = QApplication([])
window = Ui(config_fname)
app.exec()
