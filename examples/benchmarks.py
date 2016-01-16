from pytocl import clify, CLArgInfo, CLArgType
from testcode import matrix_mult, nn_layer
import pyopencl as cl
import numpy as np
from time import clock
import sys

class Benchmark:
    """A simple class to benchmark the runtimes of functions

    A new function can be added with add_func. All benchmarks can be ran by calling run
    """

    def __init__(self):
        self.benchmarks = []

    def run(self, repeat=100):
        """Runs all benchmarks available and returns how long they took

        Keyword arguments:
        times -- how often to run the benchmark (default: 100)
        """

        times = [self._get_runtime(b, repeat) for b in self.benchmarks]
        rel_times = [t / times[0] for t in times]
        
        return times, rel_times

    def add_func(self, func):
        """Adds a function to the list

        Keyword arguments:
        func -- a function to be benchmarked with no parameters
        """

        self.benchmarks.append(func)

    def _get_runtime(self, func, repeat=100):
        start_time = clock()

        for i in range(repeat):
            func()

        return clock() - start_time

def get_cl_mat_mult(shape_a, shape_b, device_type):
    shape_output = (shape_a[0], shape_b[1])

    shape_a_size = shape_a[0] * shape_a[1]
    shape_b_size = shape_b[0] * shape_b[1]
    shape_output_size = shape_output[0] * shape_output[1]

    np.random.seed(123)
    a = np.random.rand(*shape_a).astype(np.float32).flatten()
    b = np.random.rand(*shape_b).astype(np.float32).flatten()
    output = np.zeros((shape_a[0], shape_b[1])).astype(np.float32).flatten()

    arg_info = [
        CLArgInfo(CLArgType.float32_array, array_size = shape_a_size), # a
        CLArgInfo(CLArgType.float32_array, array_size = shape_b_size), # b
        CLArgInfo(CLArgType.int32), # r_a
        CLArgInfo(CLArgType.int32), # r_b
        CLArgInfo(CLArgType.float32_array, array_size = shape_output_size, is_output = True) # float* output
    ]

    func_clified = clify(matrix_mult, shape_output, arg_info, cl.Context(cl.get_platforms()[0].get_devices(device_type)))

    def cl_func():
        func_clified(a, b, shape_a[0], shape_b[0], output)

    return cl_func

def get_np_mat_mult(shape_a, shape_b):
    shape_output = (shape_a[0], shape_b[1])

    shape_a_size = shape_a[0] * shape_a[1]
    shape_b_size = shape_b[0] * shape_b[1]
    shape_output_size = shape_output[0] * shape_output[1]

    np.random.seed(123)
    a = np.random.rand(*shape_a).astype(np.float32)
    b = np.random.rand(*shape_b).astype(np.float32)
    output = np.zeros((shape_a[0], shape_b[1])).astype(np.float32)

    def np_func():
        output = np.dot(a, b)

    return np_func

def get_cl_nn_layer(shape_input, shape_weights, device_type):
    shape_output = (shape_input[0], shape_weights[1])

    shape_input_size = shape_input[0] * shape_input[1]
    shape_weights_size = shape_weights[0] * shape_weights[1]
    shape_output_size = shape_output[0] * shape_output[1]

    np.random.seed(123)
    input = np.random.rand(*shape_input).astype(np.float32).flatten()
    weights = np.random.rand(*shape_weights).astype(np.float32).flatten()
    bias = np.float32(np.random.rand())
    output = np.zeros((shape_input[0], shape_weights[1])).astype(np.float32).flatten()

    arg_info = [
        CLArgInfo(CLArgType.float32_array, array_size = shape_input_size), # input
        CLArgInfo(CLArgType.float32_array, array_size = shape_weights_size), # weights
        CLArgInfo(CLArgType.int32), # r_input
        CLArgInfo(CLArgType.int32), # r_weights
        CLArgInfo(CLArgType.float32), # bias
        CLArgInfo(CLArgType.float32_array, array_size = shape_output_size, is_output = True) # float* output
    ]

    func_clified = clify(nn_layer, shape_output, arg_info, cl.Context(cl.get_platforms()[0].get_devices(device_type)))

    # Copy the weight matrix only once (like in a realistic scenario)
    copied = False
    def cl_func():
        nonlocal copied
        func_clified(input, None if copied else weights, shape_input[0], shape_weights[0], bias, output)
        copied = True

    return cl_func

def get_np_nn_layer(shape_input, shape_weights):
    shape_output = (shape_input[0], shape_weights[1])

    shape_input_size = shape_input[0] * shape_input[1]
    shape_weights_size = shape_weights[0] * shape_weights[1]
    shape_output_size = shape_output[0] * shape_output[1]

    np.random.seed(123)
    input = np.random.rand(*shape_input).astype(np.float32)
    weights = np.random.rand(*shape_weights).astype(np.float32)
    bias = np.float32(np.random.rand())
    output = np.zeros((shape_input[0], shape_weights[1])).astype(np.float32)

    def np_func():
        output = 1.0 / (1.0 + np.exp(-(np.dot(input, weights) + bias)))

    return np_func
    
if __name__ == "__main__":
    repeat = 100

    def print_header(description):
        """ Prints the table header """
        print("##" + description)
        print("")
        print("Matrix size | Runtime Numpy | Runtime OpenCL GPU | Relative Numpy | Relative OpenCL GPU")
        print(" | ".join(["------"]*5))

    def print_row(shape, times, rel_times):
        """ Prints a benchmark table row """
        print(" | ".join([str(shape)] + ["{0:.2f}s".format(t) for t in times] + ["{0:.2f}%".format(100.0*rt) for rt in rel_times]))

    def get_times(*funcs):
        benchmark = Benchmark()
        for func in funcs:
            benchmark.add_func(func)
        return benchmark.run(repeat)

    # Benchmark Matrix Multiplication
    print_header("Matrix Multiply %s times for matrices of same size" % repeat)
    for i in range(7, 12):
        size = 2 ** i
        shape = (size, size)
        times, rel_times = get_times(get_np_mat_mult(shape, shape), get_cl_mat_mult(shape, shape, cl.device_type.GPU))
        print_row(shape, times, rel_times)

    # Benchmark Sigmoid Neural Network Layer
    print("")
    print_header("Neural Network sigmoid layer %s times for input and weight matrices of same size" % repeat)
    for i in range(7, 12):
        size = 2 ** i
        shape = (size, size)
        times, rel_times = get_times(get_np_nn_layer(shape, shape), get_cl_nn_layer(shape, shape, cl.device_type.GPU))
        print_row(shape, times, rel_times)