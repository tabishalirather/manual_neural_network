import numpy as np
from cost_fxns import *
from backprop import *
from activation_fxns import *
from forward_pass import *
from update_params import *
from data import *


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

y_full = labels()
input_full = inputs()

print("(y, input)", (y_full, input_full))
y = y_full[1]
input = input_full[1]
cost_list = list()
grad_cost_list = list()
print(network_str[1][0])


# shape_input = 4

weight_hidden, weight_output, bias_hidden, bias_output = initialize_network(
    network_str,
    num_layers=3
)


def one_pass(input, weight_hidden, weight_output, bias_hidden, bias_output, sigmoid, sigmoid_derivative, quad_cost_fxn, y):

    # Forward pass
    z1, a1, z2, a2, activations_list, zs_list, sigmoid_derivative_list, cost_fxn, grad_cost_fxn = forward_pass(input,
                                                                                                               weight_hidden,
                                                                                                               weight_output,
                                                                                                               bias_hidden,
                                                                                                               bias_output,
                                                                                                               sigmoid,
                                                                                                               sigmoid_derivative,
                                                                                                               quad_cost_fxn, y)
    # Update cost_fxn
    # store_and_print_costs(
    # cost_fxn, grad_cost_fxn, cost_list, grad_cost_list)

    # backward_pass
    delta_output_layer, delta_hidden_layer, dC_db_hidden, dC_db_output, dC_dw_hidden, dC_dw_output = do_backprop(
        grad_cost_fxn, sigmoid_derivative_list, weight_output, activations_list)

    # Update parameters
    neta = 0.5
    weight_hidden, weight_output, bias_hidden, bias_output = update_params(
        neta, weight_hidden, weight_output, dC_dw_hidden, dC_dw_output, dC_db_hidden, dC_db_output, bias_hidden, bias_output)
    # print(weight_hidden)
    store_and_print_costs(
        cost_fxn, grad_cost_fxn, cost_list, grad_cost_list)
    return [weight_hidden, weight_output, bias_hidden, bias_output, a2]


# outer loop to train the model on data multiple times
num_epochs = 100
for J in range(num_epochs):
    for i in range(len(input_full)):
        # for j in range(i, len(input_full)):
        # now we want to do one_pass (forawrd, backprop, update, cost_list) for each training example, like this
        weight_hidden, weight_output, bias_hidden, bias_output, network_output = one_pass(input_full[i], weight_hidden, weight_output, bias_hidden,
                                                                                          bias_output, sigmoid, sigmoid_derivative, quad_cost_fxn, y_full[i])
        # network_output_list.append(network_output)


network_output_list = list()

for j in range(len(input_full)):
    z1, a1, z2, a2, activations_list, zs_list, sigmoid_derivative_list, cost_fxn, grad_cost_fxn = forward_pass(input_full[j],
                                                                                                               weight_hidden,
                                                                                                               weight_output,
                                                                                                               bias_hidden,
                                                                                                               bias_output,
                                                                                                               sigmoid,
                                                                                                               sigmoid_derivative,
                                                                                                               quad_cost_fxn, y_full[j])

    network_output_list.append(a2)


def convert_one_hot_to_digit(network_output_list, y_full):
    for i in range(len(network_output_list)):
        max_index = np.argmax(network_output_list[i])
        # network_output = np.round(network_output).astype(int)
        network_output_list[i] = np.zeros_like(
            network_output_list[i]).astype(int)
        network_output_list[i][max_index] = 1
    print(network_output_list)
    y_full = np.array(y_full)
    print(y_full)
    matches = (network_output_list == y_full)
    accuracy = matches.mean()
    print("(y, network_output)", (y_full, network_output_list))
    print("Accuracy is: ", accuracy)


# for i in range(len(network_output_list)):
#     print("getting here")
convert_one_hot_to_digit(network_output_list, y_full)
# print("(y, network_output)", y_full[i], network_output_list[i])

# matches = (NET == ground_truth)
