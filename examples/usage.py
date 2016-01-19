from pytocl import CLArgType, CLArgDesc, CLFuncDesc, CLFunc, get_global_id
import numpy as np

def parallel_add(a, b, output):
    i = get_global_id(0)
    output[i] = a[i] + b[i]

# Our vectors will have 16 elements
array_size = 16

global_size = (array_size,)

# Create the descriptors for the arguments of the function
arg_desc_a = CLArgDesc(CLArgType.float32_array, array_size=array_size) # a
arg_desc_b = CLArgDesc(CLArgType.float32_array, array_size=array_size) # b
arg_desc_output = CLArgDesc(CLArgType.float32_array, array_size=array_size) # output

"""
Create the function descriptor with the global id / dimension shape information.

Arguments can be added by chaining .arg() calls (the argument order has to match the original
function's argument order (ie. arg_desc_a -> a, arg_desc_b -> b, arg_desc_output -> output).
is_readonly has to be set to False for arguments that are assigned to in the function.

copy_in() or copy_out() can be called to copy the last added argument from host to 
device before execution or from device to host after execution.
"""

func_desc = (CLFuncDesc(parallel_add, global_size)
            .arg(arg_desc_a).copy_in()
            .arg(arg_desc_b).copy_in()
            .arg(arg_desc_output, is_readonly=False).copy_out())

# Compile the actual function, you can also pass an argument to compile with a CL context.
# By default it uses cl.create_some_context()
parallel_add_cl = CLFunc(func_desc).compile()

# Create the host buffers / vectors, the dtype needs to match the arg type of the arg desc
a = np.array([ 1 ] * array_size, dtype=np.float32)
b = np.array([ 2 ] * array_size, dtype=np.float32)
output = np.array([ 0 ] * array_size, dtype=np.float32)

# Now we can execute the compiled function, we need to provide buffers for all output copies.
# For input copies we could also pass None to not copy them
parallel_add_cl(a, b, output)

# output should now be a + b

print("A:", a)
print("B:", b)
print("Output:", output)
