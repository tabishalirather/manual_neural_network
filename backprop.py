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


def do_backprop(grad_cost_fxn_output_layer, sigmoid_derivative_list, weight_output, activations_list):

    delta_output_layer = bp1(grad_cost_fxn_output_layer,
                             sigmoid_derivative_list[-1])

    print("delta_output_layer_bp1 is: ")
    print(delta_output_layer)

    delta_hidden_layer = bp2(
        weight_output, delta_output_layer, sigmoid_derivative_list[-2])
    print("delta_hidden_layer_bp2 is: ")
    print(delta_hidden_layer)

    all_delta = [delta_hidden_layer, delta_output_layer]

    dC_db = bp3(all_delta)  # hidden layer then output layer structure
    print("dC_db")
    print(dC_db)
    print(dC_db[1].shape)
    print(dC_db[2].shape)
    dC_db_hidden = dC_db[1]
    dC_db_output = dC_db[2]

    dC_dw = bp4(activations_list, all_delta)
    print("dC_dw")
    print(dC_dw)
    print(dC_dw[1].shape)
    print(dC_dw[2].shape)

    dC_dw_hidden = dC_dw[1]
    dC_dw_output = dC_dw[2]
    print("dC_dw_hidden")
    print(dC_dw_hidden)
    print(dC_dw_hidden.shape)

    print("dC_dw_output")
    print(dC_dw_output)
    print(dC_dw_output.shape)

    return [delta_output_layer, delta_hidden_layer, dC_db_hidden, dC_db_output, dC_dw_hidden, dC_dw_output]
