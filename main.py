import numpy as np
from cost_fxns import *
from backprop import *
from activation_fxns import *
from forward_pass import *
from update_params import *


def initialize_network(network_str, num_layers):
    rng = np.random.default_rng()
# shape_hidden = (3, 4)
    weight_hidden = rng.normal(loc=0, scale=1.0, size=network_str[1])
    print("weight_hidden W1, shape (3, 4):\n", weight_hidden)


# shape_output = (2, 3)
    weight_output = rng.normal(loc=0, scale=1.0, size=network_str[2])
    print("weight_output W2, shape (2, 3):\n", weight_output)
    print(network_str)

    shape_bias_hidden = network_str[1][0]
    bias_hidden = rng.normal(loc=0, scale=1.0, size=shape_bias_hidden)
    print("bias_hidden b1, shape (3,):", bias_hidden)

    shape_bias_output = network_str[2][0]
    bias_output = rng.normal(loc=0, scale=1.0, size=shape_bias_output)
    print("bias_output b2, shape (2,):", bias_output)

    return [weight_hidden, weight_output, bias_hidden, bias_output]


print("hello-hand-nn-manual")

print("input x, shape (4,):", input)

network_str = {
    0: 4,  # input
    1: (3, 4),  # hidden 1
    2: (2, 3),  # output
}

costs_lists = list()
print(network_str[1][0])
input = np.ones(network_str[0])

# shape_input = 4

weight_hidden, weight_output, bias_hidden, bias_output = initialize_network(
    network_str,
    num_layers=3
)


y = [0, 1]


# Forward pass
z1, a1, z2, a2, activations_list, zs_list, sigmoid_derivative_list, cost_fxn, grad_cost_fxn = compute_singals(input,
                                                                                                              weight_hidden,
                                                                                                              weight_output,
                                                                                                              bias_hidden,
                                                                                                              bias_output,
                                                                                                              sigmoid,
                                                                                                              sigmoid_derivative,
                                                                                                              quad_cost_fxn)


# Update cost_fxn
grad_cost_fxn_output_layer, costs_list = do_cost_stuff(
    grad_cost_fxn, y, a2, cost_fxn, costs_lists)


# backward_pass
delta_output_layer, delta_hidden_layer, dC_db_hidden, dC_db_output, dC_dw_hidden, dC_dw_output = do_backprop(
    grad_cost_fxn_output_layer, sigmoid_derivative_list, weight_output, activations_list)


# Update parameters
neta = 1
weight_hidden, weight_output, bias_hidden, bias_output = update_params(
    neta, weight_hidden, weight_output, dC_dw_hidden, dC_dw_output, dC_db_hidden, dC_db_output, bias_hidden, bias_output)
print(weight_hidden)


print("---second_pass----"*30)


# forward pass again
z1, a1, z2, a2, activations_list, zs_list, sigmoid_derivative_list, cost_fxn, grad_cost_fxn = compute_singals(input,
                                                                                                              weight_hidden,
                                                                                                              weight_output,
                                                                                                              bias_hidden,
                                                                                                              bias_output,
                                                                                                              sigmoid,
                                                                                                              sigmoid_derivative,
                                                                                                              quad_cost_fxn)


# def update_cost():
#     cost_fxn, grad_cost_fxn = quad_cost_fxn()
#     costs_list.append(cost_fxn(y, a2))
#     print(costs_list)


do_cost_stuff(
    grad_cost_fxn, y, a2, cost_fxn, costs_list)
"""



 # repeated code: *****************************************************************
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
 # now the weight update is done like this
 # w = w - neta dC_dw
 # b = b - neta dC_db
 # list_weight_output = weight_output.flatten()
 # list_weight_hidden = weight_hidden.flatten()
 # weight_list = np.concat((list_weight_hidden, list_weight_output))

 # print("weight_list")
 # print(weight_list)
 # print(weight_list.shape)

 neta = 4
 # weight_list = weight_list - neta * dC_dw_list
 # print("Updated weight list")

 weight_hidden = weight_hidden - neta * dC_dw_hidden
 weight_output = weight_output - neta * dC_dw_output

 # print(weight_list)
 print("updated weight_hidden")
 print(weight_hidden)
 print("updated weight_output")
 print(weight_output)

 dC_db_list = np.concat((dC_db[1], dC_db[2]))

 bias_list = np.concat((bias_hidden, bias_output))
 print("bias_list")
 print(bias_list)
 print(bias_list.shape)

 bias_list = bias_list - dC_db_list
 print("Updated bias list")
 print(bias_list)

 z1 = np.dot(weight_hidden, input) + bias_hidden
 print("z1 = W1 x + b1:", z1)

 a1 = sigmoid(z1)
 print("a1 = sigmoid(z1):", a1)


 z2 = np.dot(weight_output, a1) + bias_output
 print("z2 = W2 a1 + b2:", z2)

 a2 = sigmoid(z2)
 print("a2 = sigmoid(z2):", a2)

 cost_fxn, grad_cost_fxn = quad_cost_fxn()
 costs_list.append(cost_fxn(y, a2))

 print(costs_list)
 print("Decrease in cost is: ", costs_list[0] - costs_list[1])

 """
