import numpy as np


def forward_pass(input, weight_hidden, weight_output, bias_hidden, bias_output, sigmoid,  sigmoid_derivative, quad_cost_fxn, y):

    z1 = np.dot(weight_hidden, input) + bias_hidden
    print("z1 = W1 x + b1:", z1)

    a1 = sigmoid(z1)
    print("a1 = sigmoid(z1):", a1)

    z2 = np.dot(weight_output, a1) + bias_output
    print("z2 = W2 a1 + b2:", z2)

    a2 = sigmoid(z2)
    print("a2 = sigmoid(z2):", a2)

    activations_list = [input, a1, a2]
    zs_list = [z1, z2]
    sigmoid_derivative_list = [sigmoid_derivative(z1), sigmoid_derivative(z2)]
    cost_fxn, grad_cost_fxn = quad_cost_fxn(y, activations_list[-1])
#     print("grad cost fxn in forward pass is:", grad_cost_fxn)
    return [z1, a1, z2, a2, activations_list, zs_list, sigmoid_derivative_list, cost_fxn, grad_cost_fxn]
