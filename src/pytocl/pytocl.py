from .descriptors import CLArgType, CLArgDesc, CLFuncDesc
from .converter import func_to_kernel
import pyopencl as cl
import numpy as np

class CLFunc:
    """A list of CLFuncDescs which will get called sequentially and can
    share CLArgDescs. Can be compiled into a callable function using the
    compile() method.

    Instance methods:
    compile -- compiles the CLFunc into a callable function
    """

    def __init__(self, *func_descs):
        """Initializes the CL function

        Keyword arguments:
        func_descs -- the function descriptors which will get called sequentially
        """

        self.func_descs = func_descs

    def compile(self, context=cl.create_some_context(False)):
        """Compiles the function and returns the clified function

        Keyword arguments:
        context -- the CL context to use (default: cl.create_some_context(False))
        """

        # Collect all argument descriptors in the functions and add them 
        # to lists depending on whether they are used as inputs or outputs

        all_args = []
        arg_used_as_input = []
        arg_used_as_output = []

        for func_desc in self.func_descs:
            for arg_desc in func_desc.arg_descs:
                is_output = func_desc.is_output[arg_desc]

                if not is_output and not arg_desc in arg_used_as_input:
                    arg_used_as_input.append(arg_desc)

                if is_output and not arg_desc in arg_used_as_output:
                    arg_used_as_output.append(arg_desc)

                if not arg_desc in all_args:
                    all_args.append(arg_desc)

        # Create the CL Buffers for each arguments
        # used only as input: READ_ONLY
        # used only as output: WRITE_ONLY
        # used as both input and output: READ_WRITE
        # scalar parameters get None for their buffers and get passed directly

        buffers = {}

        def get_arg_desc_mem_flag(arg_desc):
            if arg_desc in arg_used_as_input and arg_desc in arg_used_as_output:
                return cl.mem_flags.READ_WRITE
            elif arg_desc in arg_used_as_input:
                return cl.mem_flags.READ_ONLY
            elif arg_desc in arg_used_as_output:
                return cl.mem_flags.WRITE_ONLY
            raise Exception("Argument neither in inputs nor in outputs")
        
        for arg_desc in all_args:
            if not CLArgType.is_array(arg_desc.arg_type):
                buffers[arg_desc] = None
            else:
                buffers[arg_desc] = cl.Buffer(context, get_arg_desc_mem_flag(arg_desc), arg_desc.byte_size)
    
        # Generate the kernels and compile them, only generate each function name once
        # TODO: Fix potential conflict with two functions with same names

        kernels = {}
        for func_desc in self.func_descs:
            if not func_desc.func_name in kernels.keys():
                kernels[func_desc.func_name] = func_to_kernel(func_desc)
        program = cl.Program(context, "\n".join(kernels.values())).build()

        # Create the queue used to enqueue copies
        queue = cl.CommandQueue(context)

        def cl_func(args):
            """Executes the clified function

            Keyword arguments:
            args -- A dictionary containing numpy arrays for each CLArgDesc 
                    used as a copy input or copy output. 
                    Input copies can be missing or None if they are arrays so they won't get copied.
            """

            def get_arg_for_desc(arg_desc):
                return args[all_args.index(arg_desc)]

            for func_desc in self.func_descs:
                # Copy inputs
                for arg_desc in func_desc.copy_in_args:
                    # Allow not passing args for input-copies or passing None, those wont get copied.
                    # If a buffer is none it means the argument is a scalar and is used directly
                    if arg_desc in args.keys() and args[arg_desc] is not None and buffers[arg_desc] is not None:
                            cl.enqueue_copy(queue, buffers[arg_desc], args[arg_desc])

                # Create the parameter list for the function
                cl_args = []
                for arg_desc in func_desc.arg_descs:
                    buffer = buffers[arg_desc]

                    # Use scalar arguments (buffer=None) directly, convert to ndarray if needed
                    if buffer is None:
                        arg = args[arg_desc]
                        cl_args.append(arg if isinstance(arg, np.ndarray) else np.array(arg))
                    else:
                        cl_args.append(buffer)

                # Execute
                prog_func = getattr(program, func_desc.func_name)
                prog_func(queue, func_desc.dim, None, *cl_args)

                # Copy outputs
                for arg_desc in func_desc.copy_out_args:
                    cl.enqueue_copy(queue, args[arg_desc], buffers[arg_desc])

        return cl_func
