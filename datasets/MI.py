from torch.utils.data import Dataset
import scipy.io
import os
import os.path
import numpy as np
import math
import datasets.config as config


class MI(Dataset):

    def __init__(self, data_type='train', cfg=config.default, ms=None, transform=None, target_transform=None):
        self.data_type = data_type
        self.cfg = cfg
        self.ms = ms
        self.transform = transform
        self.target_transform = target_transform

        if self.cfg.sMRIs.slide:
            self.ot = self.cfg.ASL.time
        else:
            self.ot = math.floor((self.cfg.sMRIs.time - self.cfg.kernel.kT) / self.cfg.kernel.dT + 1)
        self.ow = math.floor((self.cfg.sMRIs.width - self.cfg.kernel.kW) / self.cfg.kernel.dW + 1)
        self.oh = math.floor((self.cfg.sMRIs.height - self.cfg.kernel.kH) / self.cfg.kernel.dH + 1)
        self.okT = math.floor(self.cfg.ASL.time * self.cfg.ASL.overlapping / (self.ot + self.cfg.ASL.overlapping - 1))
        self.okW = math.floor(self.cfg.ASL.width * self.cfg.ASL.overlapping / (self.ow + self.cfg.ASL.overlapping - 1))
        self.okH = math.floor(self.cfg.ASL.height * self.cfg.ASL.overlapping / (self.oh + self.cfg.ASL.overlapping - 1))
        if self.cfg.sMRIs.slide:
            self.odT = 1
        else:
            self.odT = math.floor(self.okT / self.cfg.ASL.overlapping)
        self.odW = math.floor(self.okW / self.cfg.ASL.overlapping)
        self.odH = math.floor(self.okH / self.cfg.ASL.overlapping)

        self.os = self.ot * self.ow * self.oh
        self.om = self.ow * self.oh

        if self.cfg.cross_count < 2 or self.cfg.cross_count > self.cfg.data_count:
            self.cfg.cross_count = config.default.cross_count

        if self.cfg.cross_index < 1 or self.cfg.cross_index > self.cfg.cross_count:
            self.cfg.cross_index = config.default.cross_index

        self.fold_length = math.floor(self.cfg.data_count / self.cfg.cross_count)
        self.fold_start = (self.cfg.cross_index - 1) * self.fold_length + 1
        more_count = self.cfg.data_count - self.cfg.cross_count * self.fold_length
        if more_count != 0:
            adding_step = more_count / self.cfg.cross_count
            adding_count = math.floor((self.cfg.cross_index - 1) * adding_step)
            self.fold_start = self.fold_start + adding_count
            self.fold_length = self.fold_length + math.floor(self.cfg.cross_index * adding_step) - adding_count

        if self.data_type == 'train':
            self.train_data = []
            self.train_labels = []

            for i in list(range(1, self.fold_start)) + list(range(self.fold_start + self.fold_length,
                                                                  self.cfg.data_count + 1)):
                data_file = os.path.join(self.cfg.root_folder, self.cfg.paths.data_folder, self.cfg.paths.mat_folder,
                                         self.cfg.paths.sMRIs_file + str(i) + self.cfg.paths.file_type)
                if not self.cfg.ci:
                    self.train_data.append((scipy.io.loadmat(data_file))[self.cfg.paths.sMRIs_file[:-1]][0])
                else:
                    self.train_data.append(np.fromfunction(lambda i, t, w, h: np.random.rand() * \
                         (t - self.cfg.sMRIs.time / 2) ** 2 + np.random.rand() * (w - self.cfg.sMRIs.width / 2) ** 2 + \
                         np.random.rand() * (h - self.cfg.sMRIs.height / 2) ** 2 \
                         + np.random.rand() * (t - self.cfg.sMRIs.time / 2) * (w - self.cfg.sMRIs.width / 2) \
                         + np.random.rand() * (w - self.cfg.sMRIs.width / 2) * (h - self.cfg.sMRIs.height / 2) \
                         + np.random.rand() * (h - self.cfg.sMRIs.height / 2) * (t - self.cfg.sMRIs.time / 2) \
                         + np.random.rand() * 256, (2, self.cfg.sMRIs.time, self.cfg.sMRIs.width, self.cfg.sMRIs.height)))
                labels_file = os.path.join(self.cfg.root_folder, self.cfg.paths.data_folder, self.cfg.paths.mat_folder,
                                           self.cfg.paths.ASL_file + str(i) + self.cfg.paths.file_type)
                if not self.cfg.ci:
                    self.train_labels.append((scipy.io.loadmat(labels_file))[self.cfg.paths.ASL_file[:-1]][0])
                else:
                    self.train_labels.append(np.fromfunction(lambda i, t, w, h: np.random.rand() * \
                         ( t - self.cfg.ASL.time / 2) ** 2 + np.random.rand() * (w - self.cfg.ASL.width / 2) ** 2 + \
                         np.random.rand() * (h - self.cfg.ASL.height / 2) ** 2 \
                         + np.random.rand() * (t - self.cfg.ASL.time / 2) * (w - self.cfg.ASL.width / 2) \
                         + np.random.rand() * (w - self.cfg.ASL.width / 2) * (h - self.cfg.ASL.height / 2) \
                         + np.random.rand() * (h - self.cfg.ASL.height / 2) * (t - self.cfg.ASL.time / 2) \
                         + np.random.rand() * 256, (2, self.cfg.ASL.time, self.cfg.ASL.width, self.cfg.ASL.height)))

            self.train_data = np.concatenate(self.train_data)
            self.train_labels = np.concatenate(self.train_labels)

            self.count = len(self.train_labels)

            if not self.cfg.ci:
                for i in range(0, self.count):
                    self.train_data[i] = np.reshape(self.train_data[i], (1, self.train_data[i].shape[0], \
                                                self.train_data[i].shape[1], self.train_data[i].shape[2]))
                self.train_data = np.concatenate(self.train_data)
                for i in range(0, self.count):
                    self.train_labels[i] = self.train_labels[i][:, :, :, 0]
                    self.train_labels[i] = np.reshape(self.train_labels[i], (1, self.train_labels[i].shape[0], \
                                                self.train_labels[i].shape[1], self.train_labels[i].shape[2]))
                self.train_labels = np.concatenate(self.train_labels)

            if not self.cfg.nonorm:
                self.ms = [ np.concatenate([np.zeros((1, self.cfg.sMRIs.time)), np.ones((1, self.cfg.sMRIs.time))]), \
                            np.concatenate([np.zeros((1, self.cfg.ASL.time)), np.ones((1, self.cfg.ASL.time))]) ]
                for i in range(0, self.cfg.sMRIs.time):
                    self.ms[0][0, i] = np.mean(self.train_data[:, i, :, :])
                    self.ms[0][1, i] = np.std(self.train_data[:, i, :, :])
                    self.train_data[:, i, :, :] = (self.train_data[:, i, :, :] - self.ms[0][0, i]) / self.ms[0][1, i] / 3
                for i in range(0, self.cfg.ASL.time):
                    self.ms[1][0, i] = np.mean(self.train_labels[:, i, :, :])
                    self.ms[1][1, i] = np.std(self.train_labels[:, i, :, :])
                    self.train_labels[:, i, :, :] = (self.train_labels[:, i, :, :] - self.ms[1][0, i]) / self.ms[1][1, i] / 3

            if self.cfg.is3d:
                self.train_data = np.reshape(self.train_data, (self.train_data.shape[0], 1, self.train_data.shape[1], \
                                                               self.train_data.shape[2], self.train_data.shape[3]))
                self.train_labels = np.reshape(self.train_labels, (self.train_labels.shape[0], 1, self.train_labels.shape[1], \
                                               self.train_labels.shape[2], self.train_labels.shape[3]))

        elif self.data_type == 'test':
            self.test_data = []
            self.test_labels = []

            for i in range(self.fold_start, self.fold_start + self.fold_length):
                data_file = os.path.join(self.cfg.root_folder, self.cfg.paths.data_folder, self.cfg.paths.mat_folder,
                                         self.cfg.paths.sMRIs_file + str(i) + self.cfg.paths.file_type)
                if not self.cfg.ci:
                    self.test_data.append((scipy.io.loadmat(data_file))[self.cfg.paths.sMRIs_file[:-1]][0])
                else:
                    self.test_data.append(np.fromfunction(lambda i, t, w, h: np.random.rand() * \
                         (t - self.cfg.sMRIs.time / 2) ** 2 + np.random.rand() * (w - self.cfg.sMRIs.width / 2) ** 2 + \
                         np.random.rand() * (h - self.cfg.sMRIs.height / 2) ** 2 \
                         + np.random.rand() * (t - self.cfg.sMRIs.time / 2) * (w - self.cfg.sMRIs.width / 2) \
                         + np.random.rand() * (w - self.cfg.sMRIs.width / 2) * (h - self.cfg.sMRIs.height / 2) \
                         + np.random.rand() * (h - self.cfg.sMRIs.height / 2) * (t - self.cfg.sMRIs.time / 2) \
                         + np.random.rand() * 256, (2, self.cfg.sMRIs.time, self.cfg.sMRIs.width, self.cfg.sMRIs.height)))
                labels_file = os.path.join(self.cfg.root_folder, self.cfg.paths.data_folder, self.cfg.paths.mat_folder,
                                           self.cfg.paths.ASL_file + str(i) + self.cfg.paths.file_type)
                if not self.cfg.ci:
                    self.test_labels.append((scipy.io.loadmat(labels_file))[self.cfg.paths.ASL_file[:-1]][0])
                else:
                    self.test_labels.append(np.fromfunction(lambda i, t, w, h: np.random.rand() * \
                         ( t - self.cfg.ASL.time / 2) ** 2 + np.random.rand() * (w - self.cfg.ASL.width / 2) ** 2 + \
                         np.random.rand() * (h - self.cfg.ASL.height / 2) ** 2 \
                         + np.random.rand() * (t - self.cfg.ASL.time / 2) * (w - self.cfg.ASL.width / 2) \
                         + np.random.rand() * (w - self.cfg.ASL.width / 2) * (h - self.cfg.ASL.height / 2) \
                         + np.random.rand() * (h - self.cfg.ASL.height / 2) * (t - self.cfg.ASL.time / 2) \
                         + np.random.rand() * 256, (2, self.cfg.ASL.time, self.cfg.ASL.width, self.cfg.ASL.height)))

            self.test_data = np.concatenate(self.test_data)
            self.test_labels = np.concatenate(self.test_labels)

            self.count = len(self.test_labels)

            if not self.cfg.ci:
                for i in range(0, self.count):
                    self.test_data[i] = np.reshape(self.test_data[i], (1, self.test_data[i].shape[0], \
                                                self.test_data[i].shape[1], self.test_data[i].shape[2]))
                self.test_data = np.concatenate(self.test_data)
                for i in range(0, self.count):
                    self.test_labels[i] = self.test_labels[i][:, :, :, 0]
                    self.test_labels[i] = np.reshape(self.test_labels[i], (1, self.test_labels[i].shape[0], \
                                                self.test_labels[i].shape[1], self.test_labels[i].shape[2]))
                self.test_labels = np.concatenate(self.test_labels)

            if self.cfg.noise != 0:
                self.test_data *= np.random.randn(np.size(self.test_data)).reshape(self.test_data.shape) \
                                  * self.cfg.noise / 3 + 1

            if not self.cfg.nonorm and self.ms is not None:
                for i in range(0, self.cfg.sMRIs.time):
                    self.test_data[:, i, :, :] = (self.test_data[:, i, :, :] - self.ms[0][0, i]) / self.ms[0][1, i] / 3
                for i in range(0, self.cfg.ASL.time):
                    self.test_labels[:, i, :, :] = (self.test_labels[:, i, :, :] - self.ms[1][0, i]) / self.ms[1][1, i] / 3

            if self.cfg.is3d:
                self.test_data = np.reshape(self.test_data, (self.test_data.shape[0], 1, self.test_data.shape[1], \
                                                             self.test_data.shape[2], self.test_data.shape[3]))
                self.test_labels = np.reshape(self.test_labels, (self.test_labels.shape[0], 1, self.test_labels.shape[1], \
                                              self.test_labels.shape[2], self.test_labels.shape[3]))

    def __getitem__(self, index):
        dataIndex = math.floor(index / self.os)
        volIndex = index % self.os
        tIndex = math.floor(volIndex / self.om)
        mapIndex = volIndex % self.om
        hIndex = math.floor(mapIndex / self.ow)
        wIndex = mapIndex % self.ow
        wStart = wIndex * self.cfg.kernel.dW
        wEnd = wStart + self.cfg.kernel.kW
        hStart = hIndex * self.cfg.kernel.dH
        hEnd = hStart + self.cfg.kernel.kH
        if self.cfg.sMRIs.slide:
            tStart = self.cfg.sMRIs.tBlock[tIndex] - 1
        else:
            tStart = tIndex * self.cfg.kernel.dT
        tEnd = tStart + self.cfg.kernel.kT
        owStart = wIndex * self.odW
        owEnd = owStart + self.okW
        ohStart = hIndex * self.odH
        ohEnd = ohStart + self.okH
        otStart = tIndex * self.odT
        otEnd = otStart + self.okT

        if self.data_type == 'train':
            if not self.cfg.is3d:
                img, target, index = self.train_data[dataIndex][tStart:tEnd, wStart:wEnd, hStart:hEnd], \
                                     self.train_labels[dataIndex][otStart:otEnd, owStart:owEnd, ohStart:ohEnd], \
                                     np.array([dataIndex, otStart, otEnd, owStart, owEnd, ohStart, ohEnd])
            else:
                img, target, index = self.train_data[dataIndex][0:1, tStart:tEnd, wStart:wEnd, hStart:hEnd], \
                                     self.train_labels[dataIndex][0:1, otStart:otEnd, owStart:owEnd, ohStart:ohEnd], \
                                     np.array([dataIndex, otStart, otEnd, owStart, owEnd, ohStart, ohEnd])
        elif self.data_type == 'test':
            if not self.cfg.is3d:
                img, target, index = self.test_data[dataIndex][tStart:tEnd, wStart:wEnd, hStart:hEnd], \
                                     self.test_labels[dataIndex][otStart:otEnd, owStart:owEnd, ohStart:ohEnd], \
                                     np.array([dataIndex, otStart, otEnd, owStart, owEnd, ohStart, ohEnd])
            else:
                img, target, index = self.test_data[dataIndex][0:1, tStart:tEnd, wStart:wEnd, hStart:hEnd], \
                                     self.test_labels[dataIndex][0:1, otStart:otEnd, owStart:owEnd, ohStart:ohEnd], \
                                     np.array([dataIndex, otStart, otEnd, owStart, owEnd, ohStart, ohEnd])

        #print(img.shape)
        #print(target.shape)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return self.count * self.os if not self.cfg.ci or self.count * self.os < 256 else 256


def test():
    trainset = MI(data_type='train')
    #print(trainset.ms)
    #for i in range(1, len(trainset)):
    #    dl = trainset[i]
    #    print(i, end='')
    #    print(np.array(dl[0]).shape)
    #    print(np.array(dl[1]).shape)

    testset = MI(data_type='test')
    #for i in range(1, len(testset)):
    #    dl = testset[i]
    #    print(i, end='')
    #    print(np.array(dl[0]).shape)
    #    print(np.array(dl[1]).shape)

    # data_file = os.path.join('.', Config.default.paths.data_folder, Config.default.paths.mat_folder,
    #                          Config.default.paths.sMRIs_file + str(1) + Config.default.paths.file_type)
    # data = (scipy.io.loadmat(data_file))[self.cfg.paths.sMRIs_file[:-1]][0]
    # print(data[0].shape)
    #
    # labels_file = os.path.join('.', Config.default.paths.data_folder, Config.default.paths.mat_folder,
    #                            Config.default.paths.ASL_file + str(1) + Config.default.paths.file_type)
    # labels = (scipy.io.loadmat(labels_file))[self.cfg.paths.ASL_file[:-1]][0]
    # #labels = np.concatenate(labels, axis=0)
    # print(labels[0].shape)

    pass


if __name__ == "__main__":
    test()
