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

    def __init__(self, *func_descs, included_funcs=[]):
        """Initializes the CL function

        Keyword arguments:
        func_descs -- the function descriptors which will get called sequentially
        """

        self.included_funcs = included_funcs
        self.func_descs = list(func_descs)

    def compile(self, context=cl.create_some_context(False), queue=None):
        """Compiles the function and returns the clified function

        Keyword arguments:
        context -- the CL context to use (default: cl.create_some_context(False))
        queue -- the CL queue to use, creates one with the context if None is passed (default: None)
        """

        # Collect all argument descriptors in the functions and add them 
        # to lists depending on whether they are used as inputs or outputs

        all_args = []
        arg_used_readonly = []
        arg_used_not_readonly = []

        for func_desc in self.func_descs:
            for arg_desc in func_desc.arg_descs:
                is_readonly = func_desc.is_readonly(arg_desc)

                if is_readonly and not arg_desc in arg_used_readonly:
                    arg_used_readonly.append(arg_desc)

                if not is_readonly and not arg_desc in arg_used_not_readonly:
                    arg_used_not_readonly.append(arg_desc)

                if not arg_desc in all_args:
                    all_args.append(arg_desc)

        # Create the CL Buffers for each arguments
        # used only as read-only: READ_ONLY
        # used only as not read-only: WRITE_ONLY
        # used as both: READ_WRITE
        # scalar parameters get None for their buffers and get passed directly

        buffers = {}

        def get_arg_desc_mem_flag(arg_desc):
            if arg_desc in arg_used_readonly and arg_desc in arg_used_not_readonly:
                return cl.mem_flags.READ_WRITE
            elif arg_desc in arg_used_readonly:
                return cl.mem_flags.READ_ONLY
            elif arg_desc in arg_used_not_readonly:
                return cl.mem_flags.WRITE_ONLY
            raise Exception("Argument never used")
        
        for arg_desc in all_args:
            if not CLArgType.is_array(arg_desc.arg_type) or arg_desc.no_alloc:
                buffers[arg_desc] = None
            elif func_desc.is_local(arg_desc):
                buffers[arg_desc] = cl.LocalMemory(arg_desc.byte_size)
            else:
                buffers[arg_desc] = cl.Buffer(context, get_arg_desc_mem_flag(arg_desc), arg_desc.byte_size)
    
        # Generate the kernels and compile them, only generate each function name once
        # TODO: Fix potential conflict with two functions with same names

        kernels = []
        kernel_names = []
        for func_desc in self.included_funcs + self.func_descs:
            if not func_desc.func_name in kernel_names:
                kernel_names.append(func_desc.func_name)
                kernels.append(func_to_kernel(func_desc))
        program = cl.Program(context, "\n".join(kernels)).build()

        # Create a queue if None was passed
        if queue is None:
            queue = cl.CommandQueue(context)

        def cl_func(*args):
            """Executes the clified function

            Keyword arguments:
            args -- A dictionary containing numpy arrays for each CLArgDesc 
                    used as a copy input or copy output. Can also pass CL buffers to set buffers
                    for arguments that are marked no_alloc
                    Input copies can be missing or None if they are arrays so they won't get copied.
            """

            if len(args) == 1 and isinstance(args[0], dict):
                args = args[0]
            elif len(args) > 0:
                arg_dict = {}
                for i, arg in enumerate(args):
                    arg_dict[all_args[i]] = arg
                args = arg_dict

            # Make a copy of the buffers dict so we dont change the original one
            func_buffers = buffers.copy()

            # Set the buffers directly for no_alloc arguments
            for arg_desc, x in args.items():
                if arg_desc.no_alloc and isinstance(x, cl.Buffer):
                    func_buffers[arg_desc] = x

            for func_desc in self.func_descs:
                # Copy inputs
                for arg_desc in func_desc.copy_in_args:
                    # Allow not passing args for input-copies or passing None, those wont get copied.
                    # If a buffer is none it means the argument is a scalar and is used directly
                    if arg_desc in args.keys() and args[arg_desc] is not None and func_buffers[arg_desc] is not None:
                            cl.enqueue_copy(queue, func_buffers[arg_desc], args[arg_desc])

                # Create the parameter list for the function
                cl_args = []
                for arg_desc in func_desc.arg_descs:
                    buffer = func_buffers[arg_desc]

                    # Use scalar arguments (buffer=None) directly, convert to ndarray if needed
                    if buffer is None:
                        arg = args[arg_desc]
                        cl_args.append(arg if isinstance(arg, np.ndarray) else np.array(arg))
                    else:
                        cl_args.append(buffer)

                # Execute
                prog_func = getattr(program, func_desc.func_name)
                prog_func(queue, func_desc.global_size, func_desc.local_size, *cl_args)

                # Copy outputs
                for arg_desc in func_desc.copy_out_args:
                    cl.enqueue_copy(queue, args[arg_desc], func_buffers[arg_desc])

        return cl_func
