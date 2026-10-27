import numpy as np


def quad_cost_fxn():
    def cost_C(y, a):
        cost_vec = pow(np.abs(y-a), 2)
        return 0.5 * np.sum(cost_vec)

    def grad_cost_C(y, a):
        return a-y

    return [cost_C, grad_cost_C]
