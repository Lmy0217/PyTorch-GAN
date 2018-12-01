from .MI import *


def forName(dataset_name):
    return eval(dataset_name)