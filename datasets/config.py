import os


class base(object):

    def __init__(self):

        self._space = 0

    def __eq__(self, other):

        if isinstance(other, self.__class__):
            flag = True
            for name, value in vars(self).items():
                flag &= (value == eval('other.' + name))
            return flag
        else:
            return False

    def __repr__(self):

        s = []
        sp = []
        for i in range(self._space):
            sp.append(' ')

        if self._space == 0:
            s.extend(sp)
            s.append(self.__class__.__name__)
            s.append(': ')
        s.append('{\n')
        sp2 = sp.copy()
        sp2.append('  ')

        v = list(vars(self).items())
        v.sort()

        for name, value in v:
            if name in ['_space']:
                continue
            s.extend(sp2)
            s.append(name)
            s.append(': ')
            if value.__class__.__base__ == base:
                value._space = self._space + 2
            s.append(str(value))
            s.append('\n')
        s.extend(sp)
        s.append('}')
        if self._space == 0:
            s.append('\n')

        return ''.join(s)


__ASL_FILE_MAP__ = {
    'C': 'CtrlImgs_',
    'L': 'LabelImgs_',
}


class paths(base):

    def __init__(self, data_folder='datasets', mat_folder='mat', sMRIs_file='sMRIs64_', ASL_file=__ASL_FILE_MAP__['C'],
                 file_type='_c.mat', save_folder='save', check_file='_checkpoint.pth', ms_file='_ms.pth',
                 predict_file='_predict.mat', logging_file='_logging.log'):
        super(paths, self).__init__()
        self.data_folder = data_folder
        self.mat_folder = mat_folder
        self.sMRIs_file = sMRIs_file
        self.ASL_file = ASL_file
        self.file_type = file_type
        self.save_folder = save_folder
        self.check_file = check_file
        self.ms_file = ms_file
        self.predict_file = predict_file
        self.logging_file = logging_file


class kernel(base):

    def __init__(self, kT, kW, kH, dT, dW, dH):
        super(kernel, self).__init__()
        self.kT = kT
        self.kW = kW
        self.kH = kH
        self.dT = dT
        self.dW = dW
        self.dH = dH


class sMRIs(base):

    def __init__(self, width, height, time, tBlock):
        super(sMRIs, self).__init__()
        self.width = width
        self.height = height
        self.time = time
        self.tBlock = tBlock
        self.slide = self.tBlock is not None


class ASL(base):

    def __init__(self, width, height, time, overlapping):
        super(ASL, self).__init__()
        self.width = width
        self.height = height
        self.time = time
        self.overlapping = overlapping


class Dataset(base):

    def __init__(self, kernel, sMRIs, ASL, batch_size, test_batch_size, epochs, lr, lr_decay, momentum, noise=0.,
                 is3d=False, nonorm=False, nocuda=False, seed=1, ci=False, paths=paths(),
                 root_folder=os.path.abspath('.'), data_count=10, cross_index=1, cross_count=5, data_reset=False):
        super(Dataset, self).__init__()
        self.kernel = kernel
        self.sMRIs = sMRIs
        self.ASL = ASL
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.lr = lr
        self.lr_decay = lr_decay
        self.momentum = momentum
        self.noise = noise
        self.is3d = is3d
        self.nonorm = nonorm
        self.nocuda = nocuda
        self.seed = seed
        self.ci = ci
        self.paths = paths
        self.root_folder = root_folder
        self.data_count = data_count
        self.cross_index = cross_index
        self.cross_count = cross_count
        self.data_reset = data_reset


def resnet_18():
    k = kernel(64, 64, 64, 8, 9, 13)
    M = sMRIs(192, 256, 256 - 24, [ 1, 12, 23, 33, 43, 52, 60, 67, 73, 79, 85, 91, 97, 103, 110, 118, 127, 137, 147, 158, 169 ])
    C = ASL(64, 64, 21, 2)
    dataset = Dataset(k, M, C, 96, 96, 30, 0.01, 0, 0, nonorm=True)
    return dataset


def resnet_34():
    k = kernel(64, 64, 64, 8, 9, 13)
    M = sMRIs(192, 256, 256 - 24, [ 1, 12, 23, 33, 43, 52, 60, 67, 73, 79, 85, 91, 97, 103, 110, 118, 127, 137, 147, 158, 169 ])
    C = ASL(64, 64, 21, 2)
    dataset = Dataset(k, M, C, 96, 96, 30, 0.01, 0, 0, nonorm=True)
    return dataset


def resnet_50():
    k = kernel(64, 64, 64, 8, 9, 13)
    M = sMRIs(192, 256, 256 - 24, [ 1, 12, 23, 33, 43, 52, 60, 67, 73, 79, 85, 91, 97, 103, 110, 118, 127, 137, 147, 158, 169 ])
    C = ASL(64, 64, 21, 2)
    dataset = Dataset(k, M, C, 32, 32, 30, 0.01, 0, 0, nonorm=True)
    return dataset


def resnet_101():
    k = kernel(64, 64, 64, 8, 9, 13)
    M = sMRIs(192, 256, 256 - 24, [ 1, 12, 23, 33, 43, 52, 60, 67, 73, 79, 85, 91, 97, 103, 110, 118, 127, 137, 147, 158, 169 ])
    C = ASL(64, 64, 21, 2)
    dataset = Dataset(k, M, C, 16, 16, 30, 0.01, 0, 0, nonorm=True)
    return dataset


def resnet_152():
    k = kernel(64, 64, 64, 8, 9, 13)
    M = sMRIs(192, 256, 256 - 24, [ 1, 12, 23, 33, 43, 52, 60, 67, 73, 79, 85, 91, 97, 103, 110, 118, 127, 137, 147, 158, 169 ])
    C = ASL(64, 64, 21, 2)
    dataset = Dataset(k, M, C, 16, 16, 30, 0.01, 0, 0, nonorm=True)
    return dataset


def cnn_7():
    k = kernel(12, 8, 8, 12, 6, 8)
    M = sMRIs(192, 256, 256 - 4, None)
    C = ASL(64, 64, 21, 1)
    dataset = Dataset(k, M, C, 5120, 51200, 50, 0.01, 0, 0, nonorm=True)
    return dataset


def cnn_12():
    k = kernel(12, 8, 8, 12, 6, 8)
    M = sMRIs(192, 256, 256 - 4, None)
    C = ASL(64, 64, 21, 1)
    dataset = Dataset(k, M, C, 5120, 51200, 50, 0.01, 0, 0, nonorm=True)
    return dataset


def capsule():
    k = kernel(12, 28, 28, 12, 23, 15)
    M = sMRIs(192, 256, 256 - 4, None)
    C = ASL(64, 64, 21, 1)
    dataset = Dataset(k, M, C, 128, 128, 10, 0.01, 0, 0)
    return dataset


def capsule_w():
    k = kernel(12, 28, 28, 12, 23, 15)
    M = sMRIs(192, 256, 256 - 4, None)
    C = ASL(64, 64, 21, 1)
    dataset = Dataset(k, M, C, 32, 32, 10, 0.01, 0, 0)
    return dataset


def capsule_7():
    k = kernel(12, 28, 28, 12, 23, 15)
    M = sMRIs(192, 256, 256 - 4, None)
    C = ASL(64, 64, 21, 1)
    dataset = Dataset(k, M, C, 256, 256, 10, 0.01, 0, 0)
    return dataset


def capsule_12():
    k = kernel(12, 28, 28, 12, 23, 15)
    M = sMRIs(192, 256, 256 - 4, None)
    C = ASL(64, 64, 21, 1)
    dataset = Dataset(k, M, C, 128, 128, 10, 0.01, 0, 0)
    return dataset


def capsule_3():
    k = kernel(12, 28, 28, 12, 23, 15)
    M = sMRIs(192, 256, 256 - 4, None)
    C = ASL(64, 64, 21, 1)
    dataset = Dataset(k, M, C, 128, 128, 50, 0.01, 0, 0)
    return dataset


def capsule_3w():
    k = kernel(12, 28, 28, 12, 23, 15)
    M = sMRIs(192, 256, 256 - 4, None)
    C = ASL(64, 64, 21, 1)
    dataset = Dataset(k, M, C, 192, 192, 6, 0.01, 0, 0)
    return dataset


def capsule_4():
    k = kernel(12, 28, 28, 12, 23, 15)
    M = sMRIs(192, 256, 256 - 4, None)
    C = ASL(64, 64, 21, 1)
    dataset = Dataset(k, M, C, 128, 128, 50, 0.01, 0, 0)
    return dataset


def capsule_5():
    k = kernel(12, 28, 28, 12, 23, 15)
    M = sMRIs(192, 256, 256 - 4, None)
    C = ASL(64, 64, 21, 1)
    dataset = Dataset(k, M, C, 512, 512, 10, 0.01, 0, 0)
    return dataset


def squeezenet_v10():
    k = kernel(64, 64, 64, 8, 9, 13)
    M = sMRIs(192, 256, 256 - 24, [ 1, 12, 23, 33, 43, 52, 60, 67, 73, 79, 85, 91, 97, 103, 110, 118, 127, 137, 147, 158, 169 ])
    C = ASL(64, 64, 21, 2)
    dataset = Dataset(k, M, C, 96, 96, 30, 0.01, 0, 0, nonorm=True)
    return dataset


def squeezenet_v11():
    k = kernel(64, 64, 64, 8, 9, 13)
    M = sMRIs(192, 256, 256 - 24, [ 1, 12, 23, 33, 43, 52, 60, 67, 73, 79, 85, 91, 97, 103, 110, 118, 127, 137, 147, 158, 169 ])
    C = ASL(64, 64, 21, 2)
    dataset = Dataset(k, M, C, 96, 96, 30, 0.01, 0, 0, nonorm=True)
    return dataset


def fullscale():
    k = kernel(64, 192, 256, 0, 192, 256)
    M = sMRIs(192, 256, 256 - 24, [ 1, 12, 23, 33, 43, 52, 60, 67, 73, 79, 85, 91, 97, 103, 110, 118, 127, 137, 147, 158, 169 ])
    C = ASL(64, 64, 21, 1)
    dataset = Dataset(k, M, C, 2, 2, 60, 0.01, 0, 0, nonorm=True)
    return dataset


def cnn3d_3():
    k = kernel(232, 192, 256, 232, 192, 256)
    M = sMRIs(192, 256, 256 - 24, None)
    C = ASL(64, 64, 21, 1)
    dataset = Dataset(k, M, C, 1, 1, 50, 0.01, 0, 0, is3d=True, nonorm=True)
    return dataset


def cyclegan():
    k = kernel(21, 64, 64, 21, 64, 64)
    M = sMRIs(64, 64, 21, None)
    C = ASL(64, 64, 21, 1)
    dataset = Dataset(k, M, C, 1, 1, 1000, 0.0002, 0, 0)
    return dataset


default = cyclegan()


def cfg(model=None):
    if model is not None:
        return eval("" + model + "()")
    else:
        return default


def args2dataset(args):
    k = kernel(kT=args.kT, kW=args.kW, kH=args.kH, dT=args.dT, dW=args.dW, dH=args.dH)
    M = sMRIs(width=default.sMRIs.width, height=default.sMRIs.height, time=args.MTime, tBlock=default.sMRIs.tBlock if args.slide else None)
    C = ASL(width=default.ASL.width, height=default.ASL.height, time=default.ASL.time, overlapping=args.COverlapping)
    config_paths = paths(ASL_file=__ASL_FILE_MAP__[args.ASL])
    dataset = Dataset(k, M, C, cross_index=args.cross, cross_count=args.count, data_count=default.data_count,
                      paths=config_paths, root_folder=default.root_folder, batch_size=args.batchsize,
                      test_batch_size=args.testbatchsize, epochs=args.epochs, lr=args.lr, lr_decay=args.lrdecay,
                      momentum=args.momentum, noise=args.noise, is3d=default.is3d, nonorm=default.nonorm,
                      nocuda=args.no_cuda, seed=args.seed, ci=args.ci, data_reset=args.reset)
    return dataset


if __name__ == "__main__":
    print(cfg())

