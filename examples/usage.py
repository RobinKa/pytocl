from pytocl import clify, CLArgType, CLArgInfo
import numpy as np

def parallel_add(i, a, b, output):
    output[i] = a[i] + b[i]

# Our vectors will have 16 elements
array_size = 16

dim_shape = (array_size,)

arg_info = [
    CLArgInfo(CLArgType.float32_array, array_size=array_size), # a
    CLArgInfo(CLArgType.float32_array, array_size=array_size), # b
    CLArgInfo(CLArgType.float32_array, array_size=array_size, is_output=True) # output
]

parallel_add_cl = clify(parallel_add, dim_shape, arg_info)

# The dtype needs to match the arg info datatype
a = np.array([ 1 ] * array_size, dtype=np.float32)
b = np.array([ 2 ] * array_size, dtype=np.float32)
output = np.array([ 0 ] * array_size, dtype=np.float32)

parallel_add_cl(a, b, output)

# output should now be a + b elementwise

print("A:", a)
print("B:", b)
print("Output:", output)
