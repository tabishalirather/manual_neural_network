import numpy as np
from cost_fxns import quad_cost_fxn
from backprop import *


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)


print("hello-hand-nn-manual")
shape_input = 4
input = np.ones(shape_input)
print("input x, shape (4,):", input)

rng = np.random.default_rng()
shape_hidden = (3, 4)
weight_hidden = rng.normal(loc=0, scale=1.0, size=shape_hidden)
print("weight_hidden W1, shape (3, 4):\n", weight_hidden)


shape_output = (2, 3)
weight_output = rng.normal(loc=0, scale=1.0, size=shape_output)
print("weight_output W2, shape (2, 3):\n", weight_output)

shape_bias_hidden = 3
bias_hidden = rng.normal(loc=0, scale=1.0, size=shape_bias_hidden)
print("bias_hidden b1, shape (3,):", bias_hidden)

shape_bias_output = 2
bias_output = rng.normal(loc=0, scale=1.0, size=shape_bias_output)
print("bias_output b2, shape (2,):", bias_output)

z1 = np.dot(weight_hidden, input) + bias_hidden
print("z1 = W1 x + b1:", z1)

a1 = sigmoid(z1)
print("a1 = sigmoid(z1):", a1)


z2 = np.dot(weight_output, a1) + bias_output
print("z2 = W2 a1 + b2:", z2)

a2 = sigmoid(z2)
print("a2 = sigmoid(z2):", a2)
y = [0, 1]

activations_list = [input, a1, a2]
zs_list = [z1, z2]
sigmoid_derivative_list = [sigmoid_derivative(z1), sigmoid_derivative(z2)]
cost_fxn, grad_cost_fxn = quad_cost_fxn()

grad_cost_fxn_output_layer = grad_cost_fxn(y, a2)
print(20*'--')
print("cost C = 0.5 * sum((y - a2)^2):", cost_fxn(y, a2))
print("gradient of C w.r.t. a2 (a2 - y):", grad_cost_fxn(y, a2))

costs_list = list()
costs_list.append(cost_fxn(y, a2))
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
