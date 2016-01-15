from math import exp

# Computes output = -input
def negate(dim1, input, output):
    output[dim1] = -input[dim1]

# Computes output = input! elementwise
def factorial(dim1, input, output):
    output[dim1] = 1
    for i in range(2, input[dim1]+1):
        output[dim1] *= i

# Computes output = a * b
def matrix_mult(row, col, a, b, r_a, r_b, output):
    value = 0.0
    for k in range(r_b):
        val_a = a[row + r_a * k]
        val_b = b[k + r_b * col]
        value += val_a * val_b

    output[row + r_a * col] = value

# Computes output = 1 / (1 + e^(-(input * weights + bias)))
def nn_layer(row, col, input, weights, r_input, r_weights, bias, output):
    value = bias

    for k in range(r_weights):
        val_input = input[row + r_input * k]
        val_weights = weights[k + r_weights * col]
        value += val_input * val_weights

    # Sigmoid
    value = 1.0 / (1.0 + exp(-value))

    output[row + r_input * col] = value