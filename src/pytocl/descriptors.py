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
            CLArgType.float32_array: "float*",
            CLArgType.int32: "int",
            CLArgType.int32_array: "int*",
            CLArgType.bool: "bool",
            CLArgType.bool_array: "bool*",
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

    def __init__(self, arg_type, array_size=0, no_alloc=False):
        """Initializes a CLArgDesc

        Keyword arguments:
        arg_type -- the CLArgType of the argument
        array_size -- the array size of the argument, 0 for scalars (default: 0)
        no_alloc -- whether not to allocate a buffer for this argument (default: False)
        """
        
        # Avoid mistakes by checking that scalars arent no_alloc
        if no_alloc and not CLArgType.is_array(arg_type):
            raise Exception("Argument was marked as no_alloc but was scalar (scalars dont use buffers)")

        element_size = CLArgType.get_element_byte_size(arg_type)

        self.arg_type = arg_type
        self.array_size = array_size
        self.byte_size = array_size * element_size if array_size > 0 else element_size
        self.no_alloc = no_alloc

class CLFuncDesc:
    """Descriptor for a single function call
    
    Instance methods:
    arg -- adds a CLArgDesc to the function
    local_arg -- adds a CLArgDesc to the function for local memory
    copy_in -- before execution, makes the function copy an already added CLArgDesc from the host to the device
    copy_out -- after execution, makes the function copy an already added CLArgDesc from the device to the host
    """

    def __init__(self, func, global_size, local_size=None):
        """Create a new CLFuncDesc

        Keyword arguments:
        func -- the function that will be converted
        global_size -- the global work size
        local size -- the local work size (default: None)
        """

        self.func = func
        self.func_name = func.__name__
        self.global_size = global_size
        self.local_size = local_size

        self.arg_descs = []
        self.copy_in_args = []
        self.copy_out_args = []

        self.readonly_args = []
        self.local_args = []

    def is_readonly(self, arg_desc):
        return arg_desc in self.readonly_args

    def is_local(self, arg_desc):
        return arg_desc in self.local_args

    def arg(self, arg_desc, is_readonly=True):
        """Adds an argument descriptor for the functions argument, the call order will
        and should be the same as the argument order of the original function

        Keyword arguments:
        arg_desc -- the argument descriptor
        is_readonly -- whether the argument is read-only (ie. unassignable / const)
        """

        self.arg_descs.append(arg_desc)

        if is_readonly:
            self.readonly_args.append(arg_desc)

        return self

    def local_arg(self, arg_type, array_size):
        """Adds an argument descriptor for the functions argument, the call order will
        and should be the same as the argument order of the original function
        Uses local memory

        Keyword arguments:
        arg_type -- the CLArgType of the argument
        array_size -- the size of the array
        """

        local_arg_desc = CLArgDesc(arg_type, array_size)
        self.arg_descs.append(local_arg_desc)
        self.local_args.append(local_arg_desc)
        return self

    def copy_in(self, arg_desc=None):
        """Declares an argument descriptor to be copied from host to device before the
        function has executed.

        Keyword arguments:
        arg_desc -- the argument descriptor which has to be already added by calling arg()
                    if arg_desc is None, the last added argument descriptor will be used instead (default: None)
        """

        if arg_desc is None:
            arg_desc = self.arg_descs[-1]

        if self.is_local(arg_desc):
            raise Exception("Can not copy directly from host to local memory")

        self.copy_in_args.append(arg_desc)
        return self

    def copy_out(self, arg_desc=None):
        """Declares an argument descriptor to be copied from device to host after the
        function has executed

        Keyword arguments:
        arg_desc -- the argument descriptor which has to be already added by calling arg() with is_readonly=False
                    if arg_desc is None, the last added argument descriptor will be used instead (default: None)
        """

        if arg_desc is None:
            arg_desc = self.arg_descs[-1]

        if self.is_local(arg_desc):
            raise Exception("Can not copy directly from local memory to host")

        # Prevent read-only arguments from being used as copy-to-host-outputs since that would just be unnecessary copying
        if self.is_readonly(arg_desc):
            raise Exception("Arg is marked as read-only and should not be used as an output")

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