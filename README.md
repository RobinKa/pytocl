# pytocl
A python library to seamlessly convert python functions to functions making use of OpenCL

# Setup
`python setup.py install`

# Dependencies
- Python 3
- Python libraries
  - numpy
  - pyopencl
- OpenCL compatible hardware (version 1 is enough)
- An OpenCL runtime (eg. AMDAPPSDK for AMD CPUs/GPUs)

# Usage
The complete example can be found in examples/usage.py
## 1. Creating a function to be converted
Create a function (or convert an existing one) to be parallelized. The first arguments will be interpreted as the dimensions / global ids. In this example we calculate `output = a + b` for vectors, so we will be using one dimension.

```python
def parallel_add(dim1, a, b, output):
    output[dim1] = a[dim1] + b[dim1]
```

## 2. Converting the function to use OpenCL
First we need to create the information for each argument of our function excluding the dimension parameters. For this `CLArgInfo` is used which holds information on the type (`CLArgType`) of the argument, whether the parameter is used as an output, the array size of the argument (or `0` for scalars) and the size of the argument in bytes. We also need to specify the dimensions / global ids as a tuple. If we have all this we can call `clify` to convert our function.

```python
from pytocl import clify, CLArgInfo, CLArgType

[...]

# Our vectors will have 16 elements
array_size = 16

dim_shape = (array_size,)

arg_info = [
    CLArgInfo(CLArgType.float32_array, array_size=array_size), # a
    CLArgInfo(CLArgType.float32_array, array_size=array_size), # b
    CLArgInfo(CLArgType.float32_array, array_size=array_size, is_output=True) # output
]

parallel_add_cl = clify(parallel_add, dim_shape, arg_info)
```

There's also a fourth argument to `clify` that lets you specify a CL context, the default is `pyopencl.create_some_context()`.

## 3. Calling the function
You can call the function like a normal python function now, all the copying will be done for you. The passed scalars can be normal python types but all arrays have to be numpy ndarrays.

```python
import numpy as np

[...]

# The dtype needs to match the arg info datatype
a = np.array([ 1 ] * array_size, dtype=np.float32)
b = np.array([ 2 ] * array_size, dtype=np.float32)
output = np.array([ 0 ] * array_size, dtype=np.float32)

parallel_add_cl(a, b, output)

# output should now be a + b elementwise

print("A:", a)
print("B:", b)
print("Output:", output)
```

You can also pass None for arguments meaning no data will be copied and the current content will be retained(only makes sense for non-output arguments since only inputs are copied).

# Limitations / Todo list
- Only simple python functions are supported.
  - All mathematical and logical expressions
  - If statements
  - For loops currently only support range()
  - Function calls get converted to use the same name in the kernel, but the called functions themselves aren't converted if they aren't available yet
  - Type inference for local variables is currently limited. Defaults to `float`, variables starting with `index` become `int`
  - `return` is not supported, outputs have to be passed as an argument
  - Array slices are not supported
  - List comprehensions and other python-specific constructs are not supported
- Only global memory is used for array types
- Would be nice to remove the numpy dependency and use a different or create an own buffer interface
- There is no way to communicate between multiple converted functions (to prevent copying)
- Only single functions can be converted
- The source code of the function has to be available for conversion (which is often the case)
- CUDA support?
- CLEAN UP THE CODE

# Benchmarks
in `examples/benchmarks.py`

Test hardware was an AMD Phenom II 1090T and an AMD 6970
The original functions and OpenCL with CPU are orders of magnitude slower (not shown here, you can uncomment the line in `benchmarks.py` though).
Numpy versions are compared to the clified GPU versions. 

## MatMult for square matrices of different sizes, running different amount of times
---- Times 1000
Benchmark NP MatMult (16, 16) (16, 16) 0.01433813515404891 seconds
Benchmark CL GPU MatMult (16, 16) (16, 16) 1.2153505543302292 seconds
---- Times 1000
Benchmark NP MatMult (32, 32) (32, 32) 0.03683973801139517 seconds
Benchmark CL GPU MatMult (32, 32) (32, 32) 1.2517033589464583 seconds
---- Times 1000
Benchmark NP MatMult (64, 64) (64, 64) 0.05826480739870554 seconds
Benchmark CL GPU MatMult (64, 64) (64, 64) 1.300114120649412 seconds
---- Times 1000
Benchmark NP MatMult (128, 128) (128, 128) 0.17525837781058762 seconds
Benchmark CL GPU MatMult (128, 128) (128, 128) 1.352880394016558 seconds
---- Times 1000
Benchmark NP MatMult (256, 256) (256, 256) 0.6169138116715951 seconds
Benchmark CL GPU MatMult (256, 256) (256, 256) 2.1810568102929278 seconds
---- Times 100
Benchmark NP MatMult (512, 512) (512, 512) 0.4627876587666879 seconds
Benchmark CL GPU MatMult (512, 512) (512, 512) 0.7033748004285467 seconds
---- Times 10
Benchmark NP MatMult (1024, 1024) (1024, 1024) 0.28274990257141575 seconds
Benchmark CL GPU MatMult (1024, 1024) (1024, 1024) 0.3567916453068758 seconds
---- Times 10
Benchmark NP MatMult (2048, 2048) (2048, 2048) 1.9066243754443661 seconds
Benchmark CL GPU MatMult (2048, 2048) (2048, 2048) 2.8222630250492724 seconds

## Sigmoid neural network layer
---- Times 1000
Benchmark NP NN Layer (16, 16) (16, 16) 0.030530137210176278 seconds
Benchmark CL GPU NN Layer (16, 16) (16, 16) 1.280196295897941 seconds
---- Times 1000
Benchmark NP NN Layer (32, 32) (32, 32) 0.08969303361181247 seconds
Benchmark CL GPU NN Layer (32, 32) (32, 32) 1.2955175867323945 seconds
---- Times 1000
Benchmark NP NN Layer (64, 64) (64, 64) 0.1607618426364219 seconds
Benchmark CL GPU NN Layer (64, 64) (64, 64) 1.322959412439289 seconds
---- Times 1000
Benchmark NP NN Layer (128, 128) (128, 128) 0.5344054900832376 seconds
Benchmark CL GPU NN Layer (128, 128) (128, 128) 1.4247456031422985 seconds
---- Times 1000
Benchmark NP NN Layer (256, 256) (256, 256) 1.9637598493663297 seconds
Benchmark CL GPU NN Layer (256, 256) (256, 256) 2.2975419361958025 seconds
---- Times 100
Benchmark NP NN Layer (512, 512) (512, 512) 2.7003654095702103 seconds
Benchmark CL GPU NN Layer (512, 512) (512, 512) 0.6785182194943751 seconds
---- Times 10
Benchmark NP NN Layer (1024, 1024) (1024, 1024) 1.2363053125467047 seconds
Benchmark CL GPU NN Layer (1024, 1024) (1024, 1024) 0.38105275949876116 seconds
---- Times 10
Benchmark NP NN Layer (2048, 2048) (2048, 2048) 5.796788069433404 seconds
Benchmark CL GPU NN Layer (2048, 2048) (2048, 2048) 2.8163665354116247 seconds

# Contributors
- Toraxxx (Developer)
