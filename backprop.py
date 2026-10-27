import numpy as np


def bp1(grad_cost_c, sigmoid_derivative_output_layer):
    delta_output = np.multiply(grad_cost_c, sigmoid_derivative_output_layer)
    return delta_output


def bp2(weights_l_pluts_1, delta_l_pluts_1, sigmoid_derivative):
    delta_l = np.multiply(
        np.dot(np.transpose(weights_l_pluts_1), delta_l_pluts_1), sigmoid_derivative)
#     print(delta_l)
    return delta_l


def bp3(all_delta):
    dC_db = {i: val for i, val in enumerate(all_delta, start=1)}

    return dC_db


def bp4(activations, deltas):
    # activation and deltas are vectors ofcourse, we return partials w.r.t weights in each layer
    dC_hidden = np.outer(deltas[-2], activations[-3])
    dC_output = np.outer(deltas[-1], activations[-2])
    all_dC = [dC_hidden, dC_output]

    dC_da = {i: val for i, val in enumerate(all_dC, start=1)}

    return dC_da
