import numpy as np


def step_fn():
    name = 'Degrau'

    def fn(u):
        if u > 0:
            return 1
        else:
            return 0

    def derivative(y):
        return 1
    return [name, fn, derivative]


def tanh_fn():
    name = 'Tangente hiperbólica'

    def fn(u):
        return (1-np.exp(-u))/(1+np.exp(-u))

    def derivative(y):
        return 0.5*(1 - (np.square(y)))

    return [name, fn, derivative]


def lgc_fn():
    name = 'Função logística'

    def fn(u):
        return 1 / (1 + np.exp(-u))

    def derivative(y):
        return y*(1-y)

    return [name, fn, derivative]
