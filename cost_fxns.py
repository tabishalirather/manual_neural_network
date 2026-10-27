import numpy as np


def quad_cost_fxn():
    def cost_C(y, a):
        cost_vec = pow(np.abs(y-a), 2)
        return 0.5 * np.sum(cost_vec)

    def grad_cost_C(y, a):
        return a-y

    return [cost_C, grad_cost_C]


def do_cost_stuff(grad_cost_fxn, y, a2, cost_fxn, costs_list):
    grad_cost_fxn_output_layer = grad_cost_fxn(y, a2)
    print(20*'--')
    print("cost C = 0.5 * sum((y - a2)^2):", cost_fxn(y, a2))
    print("gradient of C w.r.t. a2 (a2 - y):", grad_cost_fxn(y, a2))

    # costs_list = list()
    costs_list.append(grad_cost_fxn_output_layer)
    print(costs_list)
    if len(costs_list) >= 2:
        print("Difference is: ", costs_list[-1] - costs_list[-2])
    else:
        print("Cost is: ")

    return [grad_cost_fxn_output_layer, costs_list]
