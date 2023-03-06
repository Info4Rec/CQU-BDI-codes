import numpy

from models.model import GlobalNet


def get_model(opt):
    model = GlobalNet(opt)
    return model


def dot_sim(x, y):
    return numpy.dot(x, y.T)


def cosine_sim(x, y):
    x = x / numpy.expand_dims(numpy.linalg.norm(x, axis=1), 1)
    y = y / numpy.expand_dims(numpy.linalg.norm(y, axis=1), 1)
    return numpy.dot(x, y.T)
