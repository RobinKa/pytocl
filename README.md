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
  - All literals except for strings
  - If statements
  - While loops
  - For loops currently only support range()
  - Function calls get converted to use the same name in the kernel, but the called functions themselves aren't converted if they aren't available yet
  - Type inference for local variables is currently limited. Defaults to `float`, variables starting with `i_` become `int`, variables starting with `b_` become `bool`
  - `return` is not supported, outputs have to be passed as an argument
  - Array slices are not supported
  - List comprehensions and other python-specific constructs are not supported
- Only global memory is used for array types
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

##Matrix Multiply 100 times for matrices of same size

Matrix size | Runtime Numpy | Runtime OpenCL GPU | Relative Numpy | Relative OpenCL GPU
------ | ------ | ------ | ------ | ------
(128, 128) | 0.02s | 0.06s | 100.00% | 394.49%
(256, 256) | 0.09s | 0.16s | 100.00% | 190.41%
(512, 512) | 0.39s | 0.73s | 100.00% | 187.50%
(1024, 1024) | 2.58s | 3.80s | 100.00% | 147.51%
(2048, 2048) | 22.35s | 27.97s | 100.00% | 125.15%

##Neural Network sigmoid layer 100 times for input and weight matrices of same size, weights are only copied once

Matrix size | Runtime Numpy | Runtime OpenCL GPU | Relative Numpy | Relative OpenCL GPU
------ | ------ | ------ | ------ | ------
(128, 128) | 0.05s | 0.05s | 100.00% | 111.94%
(256, 256) | 0.20s | 0.13s | 100.00% | 67.55%
(512, 512) | 2.58s | 0.56s | 100.00% | 21.60%
(1024, 1024) | 11.34s | 3.62s | 100.00% | 31.93%
(2048, 2048) | 60.99s | 27.26s | 100.00% | 44.70%

# Contributors
- Toraxxx (Developer)
