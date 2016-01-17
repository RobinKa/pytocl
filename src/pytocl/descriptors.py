from enum import Enum
import pyopencl as cl

class CLArgType(Enum):
    """Type for the parameters of functions"""
    float32 = 1,
    float32_array = 2,
    int32 = 3,
    int32_array = 4,
    bool = 5,
    bool_array = 6

    def get_element_byte_size(t):
        if t in [CLArgType.float32, CLArgType.float32_array, CLArgType.int32, CLArgType.int32_array, CLArgType.bool, CLArgType.bool_array]:
            return 4
        raise Exception("Unknown type")

    def is_array(t):
        return t in [CLArgType.float32_array, CLArgType.int32_array, CLArgType.bool_array]

    def get_cl_type_name(t):
        """Returns the CL type string of the parameter in the kernel (eg. float for float32)"""

        names = {
            CLArgType.float32: "float",
            CLArgType.float32_array: "global float*",
            CLArgType.int32: "int",
            CLArgType.int32_array: "global int*",
            CLArgType.bool: "bool",
            CLArgType.bool_array: "global bool*",
        }

        name = names[t]

        return name

class CLArgDesc:
    """Descriptor of a single parameter, can be shared for multiple functions
    
    Instance variables:
    arg_type -- The CLArgType of the argument
    array_size -- The array size of the argument, 0 for scalars
    byte_size -- The size the argument will use in bytes
    """

    def __init__(self, arg_type, array_size=0):
        """Initializes a CLArgDesc

        Keyword arguments:
        arg_type -- the CLArgType of the argument
        array_size -- the array size of the argument, 0 for scalars (default: 0)
        """

        element_size = CLArgType.get_element_byte_size(arg_type)

        self.arg_type = arg_type
        self.array_size = array_size
        self.byte_size = array_size * element_size if array_size > 0 else element_size

class CLFuncDesc:
    """Descriptor for a single function call
    
    Instance methods:
    arg -- adds a CLArgDesc to the function
    copy_in -- before execution, makes the function copy an already added CLArgDesc from the host to the device
    copy_in -- after execution, makes the function copy an already added CLArgDesc from the device to the host
    """

    def __init__(self, func, dim):
        self.func = func
        self.func_name = func.__name__
        self.dim = dim
        self.is_output = {}
        self.arg_descs = []
        self.copy_in_args = []
        self.copy_out_args = []

    def arg(self, arg_desc, is_output=False):
        """Adds an argument descriptor for the functions argument, the call order will
        and should be the same as the argument order of the original function

        Keyword arguments:
        arg_desc -- the argument descriptor
        is_output -- whether the argument is used as an output (not necessarily copied, eg. for auxiliaries)
        """

        self.arg_descs.append(arg_desc)
        self.is_output[arg_desc] = is_output
        return self

    def copy_in(self, arg_desc=None):
        """Declares an argument descriptor to be copied from host to device before the
        function has executed.

        Keyword arguments:
        arg_desc -- the argument descriptor which has to be already added by calling arg() with is_output=False
                    if arg_desc is None, the last added argument descriptor will be used instead (default: None)
        """

        if arg_desc is None:
            arg_desc = self.arg_descs[-1]

        if self.is_output[arg_desc]:
            raise Exception("Arg needs to be an input argument for copying in")
        self.copy_in_args.append(arg_desc)
        return self

    def copy_out(self, arg_desc=None):
        """Declares an argument descriptor to be copied from device to host after the
        function has executed

        Keyword arguments:
        arg_desc -- the argument descriptor which has to be already added by calling arg() with is_output=True
                    if arg_desc is None, the last added argument descriptor will be used instead (default: None)
        """

        if arg_desc is None:
            arg_desc = self.arg_descs[-1]

        if not self.is_output[arg_desc]:
            raise Exception("Arg needs to be an output argument for copying out")
        self.copy_out_args.append(arg_desc)
        return self


# Example usage:

# def funcF(dim, in a, out b)
# def funcG(dim, in b, out c)

# argDescA = CLArgDesc(CLArgType.float32_array, array_size=100)
# argDescB = CLArgDesc(CLArgType.float32_array, array_size=200)
# argDescC = CLArgDesc(CLArgType.float32_array, array_size=300)

# descF = CLFuncDesc(funcF, dim=(100,)).arg(argDescA, False).arg(argDescB, True).copy_in(argDescA).copy_in(argDescB)
# descG = CLFuncDesc(funcG, dim=(100,)).arg(argDescB, False).arg(argDescC, True).copy_out(argDescC)

# cl_func = CLFunc(descF, descG).compile()

# cl_func(a, b, c)