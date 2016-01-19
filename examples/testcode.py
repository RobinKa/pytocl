from pytocl import *

# Computes output = -input
def negate(input, output):
    i = get_global_id(0)
    output[dim1] = -input[i]

# Computes output = input! elementwise
def factorial(input, output):
    i = get_global_id(0)
    output[dim1] = 1
    for i in range(2, input[i]+1):
        output[dim1] *= i

# Computes output = a * b
def matrix_mult(a, b, r_a, r_b, output):
    i_row = get_global_id(0)
    i_col = get_global_id(1)

    value = 0.0
    for k in range(r_b):
        val_a = a[i_row + r_a * k]
        val_b = b[k + r_b * i_col]
        value += val_a * val_b

    output[i_row + r_a * i_col] = value

# Computes output = 1 / (1 + e^(-(input * weights + bias)))
def nn_layer(input, weights, r_input, r_weights, bias, output):
    i_row = get_global_id(0)
    i_col = get_global_id(1)

    value = bias
    for k in range(r_weights):
        val_input = input[i_row + r_input * k]
        val_weights = weights[k + r_weights * i_col]
        value += val_input * val_weights

    # Sigmoid
    value = 1.0 / (1.0 + cl_call("exp", -value))

    output[i_row + r_input * i_col] = value