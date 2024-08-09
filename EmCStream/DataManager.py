import numpy as np


class DataManager:
    X_file_name = ''
    labels_file_name = ''
    _X = None
    _labels = None
    current_index = 0
    data_length = 0

    def __init__(self, X_file_name, labels_file_name=None):
        self.X_file_name = X_file_name
        if labels_file_name:
            self.labels_file_name = labels_file_name
        self.load()

    def load(self):
        self._X = np.loadtxt(self.X_file_name, delimiter=',')
        if self.labels_file_name:
            self._labels = np.loadtxt(self.labels_file_name)
        self.current_index = 0
        self.data_length = len(self._X)

    def info(self):
        if not self._labels is None:
            print('X file name : [{}] \nlabels file name : [{}] \ndata shape : [{}] \nlabels shape : [{}]'.format(self.X_file_name, self.labels_file_name, self._X.shape, self._labels.shape))
        else:
            print('X file name : [{}] \nlabels file name : [{}] \ndata shape : [{}]'.format(self.X_file_name, self.labels_file_name, self._X.shape))

    def get_data(self, instance_count):
        #print self.current_index, instance_count, self.data_length
        if self.current_index + instance_count <= self.data_length:
            self.current_index = self.current_index + instance_count
            if not self._labels is None:
                return self._X[self.current_index-instance_count:self.current_index], self._labels[self.current_index-instance_count:self.current_index]
            else:
                return self._X[self.current_index-instance_count:self.current_index]
        else:
            return False, False

    def iterate_index(self, value):
        self.current_index += value

    def data_shape(self):
        return self._X.shape

    def force_data_length(self, p_length):
        if p_length < self.data_length:
            self.data_length = p_length
            print('data length is forced to : [{}]'.format(self.data_length))


