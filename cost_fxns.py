import numpy as np


def quad_cost_fxn(y, a):
    def cost_C(y, a):
        cost_vec = pow(np.abs(y-a), 2)

        return 0.5 * np.sum(cost_vec)

    def grad_cost_C(y, a):
        return a-y

    return [cost_C(y, a), grad_cost_C(y, a)]


def store_and_print_costs(cost_fxn, grad_cost_fxn, cost_list, grad_cost_list):

    # costs_list = list()
    cost_list.append(cost_fxn)
    grad_cost_list.append(grad_cost_fxn)
    print(cost_list)
    if len(cost_list) >= 2:
        print("Difference between first and last loss: ",
              (cost_list[0], cost_list[-1]), cost_list[0] - cost_list[-1])

    else:
        print("Cost is: ")

    # return [grad_cost_list, costs_list]
