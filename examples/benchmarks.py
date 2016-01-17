from pytocl import CLArgType, CLArgDesc, CLFuncDesc, CLFunc
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

def get_shape_size(shape):
    f = 1
    for x in shape:
        f *= x
    return f

def get_cl_context(device_type):
    return cl.Context(cl.get_platforms()[0].get_devices(device_type))

def get_cl_mat_mult(shape_a, shape_b, device_type):
    shape_output = (shape_a[0], shape_b[1])

    shape_a_size = shape_a[0] * shape_a[1]
    shape_b_size = shape_b[0] * shape_b[1]
    shape_output_size = shape_output[0] * shape_output[1]

    np.random.seed(123)
    a = np.random.rand(*shape_a).astype(np.float32).flatten()
    b = np.random.rand(*shape_b).astype(np.float32).flatten()
    output = np.zeros((shape_a[0], shape_b[1])).astype(np.float32).flatten()

    desc_a = CLArgDesc(CLArgType.float32_array, shape_a_size)
    desc_b = CLArgDesc(CLArgType.float32_array, shape_b_size)
    desc_rows_a = CLArgDesc(CLArgType.int32)
    desc_rows_b = CLArgDesc(CLArgType.int32)
    desc_output = CLArgDesc(CLArgType.float32_array, shape_output_size)

    func_desc = (CLFuncDesc(matrix_mult, shape_output)
                .arg(desc_a).copy_in() # a
                .arg(desc_b).copy_in() # b
                .arg(desc_rows_a).copy_in() # rows a
                .arg(desc_rows_b).copy_in() # rows b
                .arg(desc_output, True).copy_out()) # output

    func_clified = CLFunc(func_desc).compile(get_cl_context(device_type))

    def cl_func():
        func_clified({desc_a: a, desc_b: b, desc_rows_a: shape_a[0], desc_rows_b: shape_b[0], desc_output: output})

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

    desc_input = CLArgDesc(CLArgType.float32_array, shape_input_size)
    desc_weights = CLArgDesc(CLArgType.float32_array, shape_weights_size)
    desc_rows_input = CLArgDesc(CLArgType.int32)
    desc_rows_weights = CLArgDesc(CLArgType.int32)
    desc_bias = CLArgDesc(CLArgType.float32)
    desc_output = CLArgDesc(CLArgType.float32_array, shape_output_size)

    func_desc = (CLFuncDesc(nn_layer, shape_output)
                .arg(desc_input).copy_in() # input
                .arg(desc_weights).copy_in() # weights
                .arg(desc_rows_input).copy_in() # rows_input
                .arg(desc_rows_weights).copy_in() # rows_weights
                .arg(desc_bias).copy_in() # bias
                .arg(desc_output, True).copy_out()) # output

    func_clified = CLFunc(func_desc).compile(get_cl_context(device_type))

    # Copy the weight matrix only once (like in a realistic scenario)
    copied = False
    def cl_func():
        nonlocal copied
        func_clified({desc_input: input, desc_weights: None if copied else weights, desc_rows_input: shape_input[0], 
                      desc_rows_weights: shape_weights[0], desc_bias: bias, desc_output: output})
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


def get_cl_mlp(batch_size, input_size, shape_weights_a, shape_weights_b, device_type):
    shape_input = (batch_size, input_size)
    shape_aux_a = (batch_size, shape_weights_a[1])
    shape_output = (batch_size, shape_weights_b[1])

    np.random.seed(123)
    input = np.random.rand(*shape_input).astype(np.float32)
    weights_a = np.random.rand(*shape_weights_a).astype(np.float32)
    bias_a = np.float32(np.random.rand())
    weights_b = np.random.rand(*shape_weights_b).astype(np.float32)
    bias_b = np.float32(np.random.rand())
    output = np.zeros(shape_output).astype(np.float32)

    desc_input = CLArgDesc(CLArgType.float32_array, get_shape_size(shape_input))
    desc_weights_a = CLArgDesc(CLArgType.float32_array, get_shape_size(shape_weights_a))
    desc_rows_input = CLArgDesc(CLArgType.int32)
    desc_rows_weights_a = CLArgDesc(CLArgType.int32)
    desc_bias_a = CLArgDesc(CLArgType.float32)
    desc_aux_a = CLArgDesc(CLArgType.float32_array, get_shape_size(shape_aux_a))

    desc_weights_b = CLArgDesc(CLArgType.float32_array, get_shape_size(shape_weights_b))
    desc_rows_aux_a = CLArgDesc(CLArgType.int32)
    desc_rows_weights_b = CLArgDesc(CLArgType.int32)
    desc_bias_b = CLArgDesc(CLArgType.float32)
    desc_output = CLArgDesc(CLArgType.float32_array, get_shape_size(shape_output))

    func_desc_a = (CLFuncDesc(nn_layer, shape_aux_a)
                .arg(desc_input).copy_in()
                .arg(desc_weights_a).copy_in()
                .arg(desc_rows_input).copy_in()
                .arg(desc_rows_weights_a).copy_in()
                .arg(desc_bias_a).copy_in()
                .arg(desc_aux_a, True))

    func_desc_b = (CLFuncDesc(nn_layer, shape_output)
                .arg(desc_aux_a)
                .arg(desc_weights_b).copy_in()
                .arg(desc_rows_aux_a).copy_in()
                .arg(desc_rows_weights_b).copy_in()
                .arg(desc_bias_b).copy_in()
                .arg(desc_output, True)).copy_out()

    func_clified = CLFunc(func_desc_a, func_desc_b).compile(get_cl_context(device_type))

    # Copy the weight matrices only once (like in a realistic scenario)
    copied = False
    def cl_func():
        nonlocal copied
        func_clified({desc_input: input, desc_weights_a: None if copied else weights_a, desc_rows_input: shape_input[0], 
                             desc_rows_weights_a: shape_weights_a[0], desc_bias_a: bias_a,
                             desc_weights_b: None if copied else weights_b, desc_rows_aux_a: shape_aux_a[0], 
                             desc_rows_weights_b: shape_weights_b[0], desc_bias_b: bias_b,
                             desc_output: output})
        copied = True

    return cl_func

def get_np_mlp(batch_size, input_size, shape_weight_a, shape_weights_b):
    shape_input = (batch_size, input_size)
    shape_output = (batch_size, shape_weights_b[1])

    np.random.seed(123)
    input = np.random.rand(*shape_input).astype(np.float32)
    weights_a = np.random.rand(*shape_weight_a).astype(np.float32)
    bias_a = np.random.rand()
    weights_b = np.random.rand(*shape_weights_b).astype(np.float32)
    bias_b = np.random.rand()
    output = np.zeros(shape_output).astype(np.float32)

    def np_func():
        output = np.dot(input, weights_a) + bias_a
        output = np.dot(output, weights_b) + bias_b

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
    
    # Benchmark 2 layer NN
    print("")
    print_header("2 layer MLP with 128 batch size and input vectors / weight matrices of same size %s times" % repeat)
    for i in range(7, 12):
        size = 2 ** i

        batch_size = 128
        input_size = size
        weights_a_shape = (size, size)
        weights_b_shape = (size, size)
        
        times, rel_times = get_times(get_np_mlp(batch_size, input_size, weights_a_shape, weights_b_shape), 
                                     get_cl_mlp(batch_size, input_size, weights_a_shape, weights_b_shape, cl.device_type.GPU))

        print_row(weights_a_shape, times, rel_times)