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
The complete example can be found in examples/usage.py. For more examples check tests/cltest.py and examples/benchmarks.py.

## 1. Creating a function to be converted
Create a function (or convert an existing one) to be parallelized. The first arguments will be interpreted as the dimensions / global ids. In this example we calculate `output = a + b` for vectors, so we will be using one dimension.

```python
def parallel_add(dim1, a, b, output):
    output[dim1] = a[dim1] + b[dim1]
```

## 2. Converting the function
First we need to create the information for each argument of our function excluding the dimension parameters. For this `CLArgDesc` is used which holds information about the type of the argument as well as the `array_size` (0 for scalars) and whether the argument is used as an output (but not necessarily copied back to the host).

```python
from pytocl import CLArgType, CLArgDesc, CLFuncDesc, CLFunc

[...]

# Our vectors will have 16 elements
array_size = 16

dim_shape = (array_size,)

# Create the descriptors for the arguments of the function, excluding the dimension
arg_desc_a = CLArgDesc(CLArgType.float32_array, array_size=array_size) # a
arg_desc_b = CLArgDesc(CLArgType.float32_array, array_size=array_size) # b
arg_desc_output = CLArgDesc(CLArgType.float32_array, array_size=array_size, is_output=True) # output
```

Next we need to create the descriptor for the function itself which has the gloabl id / dimension shape information and more information about its arguments (ie. whether they're copied from host to device before execution and whether theyre copied from device to host after execution)

```python
"""
Create the function descriptor with the global id / dimension shape information.

Arguments can be added by chaining .arg() calls (the argument order has to match the original
function's argument order (ie. arg_desc_a -> a, arg_desc_b -> b, arg_desc_output -> output).
is_output has to be set to True for arguments that are used as outputs in the function (ie.
those that can't be declared constant).

copy_in() or copy_out() can be called to copy the last added argument from host to 
device before execution or from device to host after execution.
"""

func_desc = (CLFuncDesc(parallel_add, dim_shape)
            .arg(arg_desc_a).copy_in()
            .arg(arg_desc_b).copy_in()
            .arg(arg_desc_output, is_output=True).copy_out())
```

Now we can compile the function which gives us a normal python function we can call

```python
# Compile the actual function, you can also pass an argument to compile with a CL context.
# By default it uses cl.create_some_context()
parallel_add_cl = CLFunc(func_desc).compile()
```

## 3. Calling the function
You can call the function like a normal python function now, all the copying will be done for you. The passed scalars can be normal python types but all arrays have to be numpy ndarrays.

```python
import numpy as np

[...]

# Create the host buffers / vectors, the dtype needs to match the arg type of the arg desc
a = np.array([ 1 ] * array_size, dtype=np.float32)
b = np.array([ 2 ] * array_size, dtype=np.float32)
output = np.array([ 0 ] * array_size, dtype=np.float32)

# Now we can execute the compiled function, we need to provide buffers for all output copies.
# For input copies we could also pass None or leave them out to not copy them
parallel_add_cl({arg_desc_a: a, arg_desc_b: b, arg_desc_output: output})

# output should now be a + b

print("A:", a)
print("B:", b)
print("Output:", output)
```

You can also pass None for arguments only used as copy-inputs meaning no data will be copied and the current content will be retained.

# Limitations / Todo list
- Only simple python functions are supported.
  - All mathematical and logical expressions
  - All literals except for strings
  - If statements and IfElse constructs
  - While loops
  - For loops currently only support range()
  - Function calls get converted to use the same name in the kernel, but the called functions themselves aren't converted if they aren't available yet (eg. currently you can `from math import exp` and use exp(x))
  - Type inference for local variables is currently limited. Defaults to `float`, variables starting with `i_` become `int`, variables starting with `b_` become `bool`
  - `return` is not supported, outputs have to be passed as an argument
  - Array slices are not supported
  - List comprehensions and other python-specific constructs are not supported
- Only global memory is used for array types
- The source code of the function has to be available for conversion (which is often the case)
- CUDA support?
- Clean up the converter code

# Benchmarks
in `examples/benchmarks.py`

Test hardware was an AMD Phenom II 1090T and an AMD 6970
The original functions and OpenCL with CPU are orders of magnitude slower (not shown here, you can uncomment the line in `benchmarks.py` though).
Numpy versions are compared to the clified GPU versions. 

##Matrix Multiply 100 times for matrices of same size

Matrix size | Runtime Numpy | Runtime OpenCL GPU | Relative Numpy | Relative OpenCL GPU
------ | ------ | ------ | ------ | ------
(128, 128) | 0.02s | 0.08s | 100.00% | 372.00%
(256, 256) | 0.08s | 0.16s | 100.00% | 200.18%
(512, 512) | 0.40s | 0.60s | 100.00% | 148.81%
(1024, 1024) | 2.64s | 3.64s | 100.00% | 137.71%
(2048, 2048) | 17.76s | 28.07s | 100.00% | 158.03%

##Neural Network sigmoid layer 100 times for input and weight matrices of same size

Matrix size | Runtime Numpy | Runtime OpenCL GPU | Relative Numpy | Relative OpenCL GPU
------ | ------ | ------ | ------ | ------
(128, 128) | 0.05s | 0.05s | 100.00% | 114.13%
(256, 256) | 0.19s | 0.13s | 100.00% | 70.13%
(512, 512) | 2.62s | 0.54s | 100.00% | 20.62%
(1024, 1024) | 11.52s | 3.65s | 100.00% | 31.66%
(2048, 2048) | 60.01s | 27.40s | 100.00% | 45.66%

# Contributors
- Toraxxx (Developer)
