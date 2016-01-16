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

    def run(self, times=100):
        """Runs all benchmarks available and prints how long they took to run

        Keyword arguments:
        times -- how often to run the benchmark (default: 100)
        """

        times = [self.get_runtime(b[1], times) for b in self.benchmarks]
        rel_times = ["{0:.2f}%".format(100.0 * t / times[0]) for t in times]

        print(" | ".join(["{0:.2f}".format(t) + "s" for t in times] + rel_times))
            

    def add_func(self, name, func):
        """Adds a function to the list with a name that will be printed

        Keyword arguments:
        name -- the name of the benchmark
        func -- a function to be benchmarked with no parameters
        """

        self.benchmarks.append((name, func))

    def get_runtime(self, func, times=100):
        start_time = clock()

        for i in range(times):
            func()

        return clock() - start_time
        
class MatMultBenchmark(Benchmark):
    """Benchmarks matrix multiplication (c = a * b)"""

    def __init__(self, shape_a, shape_b):
        super().__init__()

        shape_output = (shape_a[0], shape_b[1])

        shape_a_size = shape_a[0] * shape_a[1]
        shape_b_size = shape_b[0] * shape_b[1]
        shape_output_size = shape_output[0] * shape_output[1]

        np.random.seed(123)
        a = np.random.rand(*shape_a).astype(np.float32)
        b = np.random.rand(*shape_b).astype(np.float32)
        output = np.zeros((shape_a[0], shape_b[1])).astype(np.float32)

        a_cl = a.flatten()
        b_cl = b.flatten()
        output_cl = output.flatten()

        # Numpy
        def np_func():
            output = np.dot(a, b)

        self.add_func("NP MatMult %s %s" % (shape_a, shape_b), np_func)

        # OpenCL CPU/GPU
        arg_info = [
            CLArgInfo(CLArgType.float32_array, array_size = shape_a_size), # a
            CLArgInfo(CLArgType.float32_array, array_size = shape_b_size), # b
            CLArgInfo(CLArgType.int32), # r_a
            CLArgInfo(CLArgType.int32), # r_b
            CLArgInfo(CLArgType.float32_array, array_size = shape_output_size, is_output = True) # float* output
        ]

        func_cpu_clified = clify(matrix_mult, shape_output, arg_info, cl.Context(cl.get_platforms()[0].get_devices(cl.device_type.CPU)))
        func_gpu_clified = clify(matrix_mult, shape_output, arg_info, cl.Context(cl.get_platforms()[0].get_devices(cl.device_type.GPU)))

        def cl_cpu_func():
            func_cpu_clified(a_cl, b_cl, shape_a[0], shape_b[0], output_cl)

        def cl_gpu_func():
            func_gpu_clified(a_cl, b_cl, shape_a[0], shape_b[0], output_cl)

        #self.add_func("CL CPU MatMult %s %s" % (shape_a, shape_b), cl_cpu_func)
        self.add_func("CL GPU MatMult %s %s" % (shape_a, shape_b), cl_gpu_func)

class NNLayerBenchmark(Benchmark):
    """Benchmarks a simple neural network layer (output = 1/(1+e^(-input*weights+bias)))"""

    def __init__(self, shape_input, shape_weights):
        super().__init__()

        shape_output = (shape_input[0], shape_weights[1])

        shape_input_size = shape_input[0] * shape_input[1]
        shape_weights_size = shape_weights[0] * shape_weights[1]
        shape_output_size = shape_output[0] * shape_output[1]

        np.random.seed(123)
        input = np.random.rand(*shape_input).astype(np.float32)
        weights = np.random.rand(*shape_weights).astype(np.float32)
        bias = np.float32(np.random.rand())
        output = np.zeros((shape_input[0], shape_weights[1])).astype(np.float32)

        input_cl = input.flatten()
        weights_cl = weights.flatten()
        output_cl = output.flatten()

        # Numpy
        def np_func():
            output = 1.0 / (1.0 + np.exp(-(np.dot(input, weights) + bias)))

        self.add_func("NP NN Layer %s %s" % (shape_input, shape_weights), np_func)

        # OpenCL CPU/GPU
        arg_info = [
            CLArgInfo(CLArgType.float32_array, array_size = shape_input_size), # input
            CLArgInfo(CLArgType.float32_array, array_size = shape_weights_size), # weights
            CLArgInfo(CLArgType.int32), # r_input
            CLArgInfo(CLArgType.int32), # r_weights
            CLArgInfo(CLArgType.float32), # bias
            CLArgInfo(CLArgType.float32_array, array_size = shape_output_size, is_output = True) # float* output
        ]

        func_cpu_clified = clify(nn_layer, shape_output, arg_info, cl.Context(cl.get_platforms()[0].get_devices(cl.device_type.CPU)))
        func_gpu_clified = clify(nn_layer, shape_output, arg_info, cl.Context(cl.get_platforms()[0].get_devices(cl.device_type.GPU)))

        # Copy the weight matrix only once (like in a realistic scenario)
        self.cpu_copied = False
        def cl_cpu_func():
            func_cpu_clified(input_cl, None if self.cpu_copied else weights_cl, shape_input[0], shape_weights[0], bias, output_cl)
            self.cpu_copied = True

        self.gpu_copied = False
        def cl_gpu_func():
            func_gpu_clified(input_cl, None if self.cpu_copied else weights_cl, shape_input[0], shape_weights[0], bias, output_cl)
            self.gpu_copied = True

        #self.add_func("CL CPU NN Layer %s %s" % (shape_input, shape_weights), cl_cpu_func)
        self.add_func("CL GPU NN Layer %s %s" % (shape_input, shape_weights), cl_gpu_func)
    
if __name__ == "__main__":
    times = 100

    print("##Matrix Multiply 100 times for matrices of same size")
    print("")
    print("Matrix size | Runtime Numpy | Runtime OpenCL GPU | Relative Numpy | Relative OpenCL GPU")
    print(" | ".join(["------"]*5))
    for i in range(7, 12):
        size = 2 ** i
        shape = (size, size)
        sys.stdout.write("%s | " % (shape,))
        MatMultBenchmark(shape, shape).run(times)
        
    print("")
    print("##Neural Network sigmoid layer 100 times for input and weight matrices of same size")
    print("")
    print("Matrix size | Runtime Numpy | Runtime OpenCL GPU | Relative Numpy | Relative OpenCL GPU")
    print(" | ".join(["------"]*5))
    for i in range(7, 12):
        size = 2 ** i
        shape = (size, size)
        sys.stdout.write("%s | " % (shape,))
        NNLayerBenchmark(shape, shape).run(times)