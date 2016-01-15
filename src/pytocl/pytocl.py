from enum import Enum
import ast
import inspect
import numpy as np
import pyopencl as cl

class CLArgType(Enum):
    """Type for the parameters of functions to convert"""

    float32 = 1,
    float32_array = 2,
    int32 = 3,
    int32_array = 4,

    def is_array(t):
        return t == CLArgType.float32_array or t == CLArgType.int32_array

class CLArgInfo():
    """Information about the parameters of functions to convert
    
    Instance variables:
    arg_type -- The CLArgType of the argument
    is_output -- Whether the argument is used as an output
    array_size -- The array size of the argument, 0 for scalars
    byte_size -- The size the argument will use in bytes

    Methods:
    get_cl_type_decl -- Returns the CL type string of the parameter in the kernel
    """

    def __init__(self, arg_type, is_output=False, array_size=0, element_size=4):
        """Initializes a CLArgInfo

        Keyword arguments:
        arg_type -- The CLArgType of the argument
        is_output -- Whether the argument is used as an output (default: False)
        array_size -- The array size of the argument, 0 for scalars (default: 0)
        element_size -- The size of one element in bytes (default: 4)

        """
        self.arg_type = arg_type
        self.is_output = is_output
        self.array_size = array_size
        self.byte_size = array_size * element_size if array_size > 0 else element_size

    def get_cl_type_decl(self):
        """Returns the CL type string of the parameter in the kernel

        Uses const for non-output parameters
        """

        strings = {
            CLArgType.float32: "float",
            CLArgType.float32_array: "global float*",
            CLArgType.int32: "int",
            CLArgType.int32_array: "global int*"
        }

        decl = strings[self.arg_type]

        if not self.is_output:
            decl = "const " + decl

        return decl

class CLVisitor(ast.NodeVisitor):
    """A visitor that visits each node in a function's AST and
    generates an OpenCL kernel as a string
    
    Methods:
    get_code -- Gets the generated kernel code
    """

    unary_ops = {
        ast.UAdd: "+",
        ast.USub: "-",
        ast.Not: "!",
        ast.Invert: "~"
    }

    binary_ops = {
        ast.Add: "+",
        ast.Sub: "-",
        ast.Mult: "*",
        ast.Div: "/",
        ast.FloorDiv: "/",
        ast.Mod: "%",
        ast.LShift: "<<",
        ast.RShift: ">>",
        ast.BitOr: "|",
        ast.BitXor: "^",
        ast.BitAnd: "&"    
    }

    compare_ops = {
        ast.Eq: "==",
        ast.NotEq: "!=",
        ast.Lt: "<",
        ast.LtE: "<=",
        ast.Gt: ">",
        ast.GtE: ">=",
        ast.Is: "==",
        ast.IsNot: "!="
    }

    def __init__(self, func_name, dim_shape, arg_info):
        """Initializes a new CLVisitor

        Keyword arguments:
        func_name -- The name of the generated function in the kernel
        dim_shape -- the dimension shape, len(dim_shape) must be between 1 and 3
        arg_info -- CLArgInfo for each argument of func, excluding the first len(dim_shape) args used for dimensions
        """
        super().__init__()

        if len(dim_shape) <= 0 or len(dim_shape) > 3:
            raise Exception("Unsupported dimension size " + str(len(dim_shape)))

        self.func_name = func_name
        self.dim_shape = dim_shape
        self.arg_info = arg_info
        
        self.indent = 0        
        self.code = []

        self.declared_vars = [[]]
        
    def is_var_declared(self, var):
        return any([var in block_vars for block_vars in self.declared_vars])

    def declare_var(self, var):
        self.declared_vars[-1].append(var)

    def push_block(self):
        self.indent += 1
        self.declared_vars.append([])

    def pop_block(self):
        self.indent -= 1
        self.declared_vars.pop()

    def get_code(self):
        """Returns the generated kernel code"""
        return "\n".join(self.code)
        
    def append(self, s):
        self.code[-1] += s

    def write(self, line):
        self.code.append("    " * self.indent + line)
        
    def visit_FunctionDef(self, node):
        if node.name != self.func_name:
            raise Exception("Node function name is not identical with passed name")
        
        func_args = [arg.arg for arg in node.args.args]

        self.write("kernel void " + self.func_name + "(")

        dim_count = len(self.dim_shape)

        # Write arguments
        # TODO: Check whether args are identical to tree args
        self.append(",".join([self.arg_info[i].get_cl_type_decl() + " " + arg for i, arg in enumerate(func_args[dim_count:])]))
        self.append(")")

        self.write("{")
        self.push_block()

        # Write dimension ints
        for i in range(len(self.dim_shape)):
            var_name = "dim%s" % (i+1)
            self.write("int " + var_name + " = get_global_id(" + str(i) + ");")
            self.declare_var(var_name)

        for arg in func_args:
            self.declare_var(arg)

        self.generic_visit(node)

        self.pop_block()

        self.write("}")

    def visit_Name(self, node):
        if not self.is_var_declared(node.id):
            self.declare_var(node.id)

            # TODO: Type inference
            type_name = "int" if node.id.startswith("index") else "float"

            self.append(type_name + " " + node.id)
        else:
            self.append(node.id)

    def visit_Num(self, node):
        s = str(node.n)
        if isinstance(node.n, float):
            if "." in s:
                s += "f"
            else:
                s += ".f"

        self.append(s)

    def visit_UnaryOp(self, node):
        op_type = type(node.op)

        self.append("(")

        if not op_type in CLVisitor.unary_ops:
            raise Exception("Unknown unary op " + str(op_type))

        self.append(CLVisitor.unary_ops[op_type])
        self.visit(node.operand)

        self.append(")")

    def visit_BoolOp(self, node):
        op = node.op
        values = node.values

        str_op = None 

        self.append("(")

        if isinstance(op, ast.And):
            str_op = "&&"
        elif isinstance(op, ast.Or):
            str_op = "||"
        else:
            raise Exception("Unsupported bool op " + str(type(op)))

        for i, value in enumerate(values):
            self.visit(value)
            if i != len(values)-1:
                self.append(str_op)

        self.append(")")

    def visit_Compare(self, node):
        values = [node.left] + node.comparators
        ops = node.ops

        self.append("(")

        for i in range(len(values)-1):
            op_type = type(ops[i])

            if not op_type in CLVisitor.compare_ops:
                raise Exception("Unsupported compare op " + str(op_type))

            self.visit(values[i])
            self.append(CLVisitor.compare_ops[op_type])
            self.visit(values[i+1])

            if i != len(values)-2:
                self.append("&&")

        self.append(")")

    def visit_BinOp(self, node):
        left = node.left
        right = node.right
        op_type = type(node.op)

        self.append("(")

        if op_type == ast.Pow:
            self.append("pow(")
            self.visit(left)
            self.append(",")
            self.visit(right)
            self.append(")")
        elif op_type in CLVisitor.binary_ops:
            self.visit(left)
            self.append(CLVisitor.binary_ops[op_type])
            self.visit(right)
        else:
            raise Exception("Unsupported bin op " + str(op_type))

        self.append(")")

    def visit_Expr(self, node):
        self.write("")
        self.generic_visit(node)
        self.append(";")

    def visit_Assign(self, node):
        targets = node.targets
        value = node.value

        for target in targets:
            self.write("")
            self.visit(target)
            self.append("=")
            self.visit(value)
            self.append(";")

    def visit_AugAssign(self, node):
        target = node.target
        op_type = type(node.op)
        value = node.value

        if op_type in CLVisitor.binary_ops:
            self.write("")
            self.visit(target)
            self.append(CLVisitor.binary_ops[op_type] + "=")
            self.visit(value)
            self.append(";")
        else:
            raise Exception("Unsupported bin op " + str(op_type))

    def visit_Subscript(self, node):
        value = node.value
        slice = node.slice
        
        if not isinstance(slice, ast.Index):
            raise Exception("Only single index subscripts are supported")

        self.visit(value)
        self.append("[")
        self.visit(slice)
        self.append("]")

    def visit_If(self, node):
        test = node.test
        bodies = node.body
        orelses = node.orelse

        self.write("if(")
        self.visit(test)
        self.append(")")
        self.write("{")
        self.push_block()
        for body in bodies:
            self.visit(body)
        self.pop_block()
        self.write("}")

        if len(orelses) > 0:
            self.write("else")
            self.write("{")
            self.push_block()
            for orelse in orelses:
                self.visit(orelse)
            self.pop_block()
            self.write("}")

    def visit_For(self, node):
        iter_var = node.target
        iter = node.iter
        body = node.body
        orelse = node.orelse # TODO

        if not isinstance(iter, ast.Call) or not isinstance(iter.func, ast.Name) or not iter.func.id == "range":
            raise Exception("Unsupported for loop:", ast.dump(node))

        if len(iter.args) == 1:
            self.write("for(int " + iter_var.id + "=0;" + iter_var.id + "<")
            self.visit(iter.args[0])
            self.append(";" + iter_var.id + "++)")
        elif len(iter.args) == 2:
            self.write("for(int " + iter_var.id + "=")
            self.visit(iter.args[0])
            self.append(";" + iter_var.id + "<")
            self.visit(iter.args[1])
            self.append(";" + iter_var.id + "++)")
        elif len(iter.args) == 3:
            self.write("for(int " + iter_var.id + "=")
            self.visit(iter.args[0])
            self.append(";" + iter_var.id + "<")
            self.visit(iter.args[1])
            self.append(";" + iter_var.id + "+=")
            self.visit(iter.args[2])
            self.append(")")

        self.write("{")

        self.push_block()

        self.declare_var(iter_var.id)

        for b in body:
            self.visit(b)

        self.pop_block()

        self.write("}")

    def visit_Call(self, node):
        func = node.func
        args = node.args
        keyword = node.keywords

        if not isinstance(func, ast.Name):
            raise Exception("Unsupported function call")

        # TODO: Check if function is supported
        self.append(func.id)

        self.append("(")

        for i, arg in enumerate(args):
            self.visit(arg)
            if i != len(args)-1:
                self.append(",")

        self.append(")")

def func_to_kernel(func, dim_shape, arg_info):
    """Converts a python function to an OpenCL kernel as a string

    Keyword arguments:
    func -- the function to convert
    dim_shape -- the dimension shape, len(dim_shape) must be between 1 and 3
    arg_info -- CLArgInfo for each argument of func, excluding the first len(dim_shape) args used for dimensions
    """

    func_name = func.__name__

    source = inspect.getsource(func)
    tree = ast.parse(source)

    visitor = CLVisitor(func_name, dim_shape, arg_info)
    visitor.visit(tree)

    return visitor.get_code()

def clify(func, dim_shape, arg_info, context=cl.create_some_context(False)):
    """Converts a python function to an OpenCL function

    eg. func(dim1, input, output) -> cl_func(input, output)

    Passing None as an argument to the returned function will not copy anything 
    for the argument and keep the already existing data for that argument

    Keyword arguments:
    func -- the function to convert
    dim_shape -- the dimension shape, len(dim_shape) must be between 1 and 3
    arg_info -- CLArgInfo for each argument of func, excluding the first len(dim_shape) args used for dimensions
    context -- the OpenCL context to execute in (default cl.create_some_context(False))
    """

    func_name = func.__name__

    # Convert the function to a kernel string
    kernel = func_to_kernel(func, dim_shape, arg_info)

    buffers = []
    
    queue = cl.CommandQueue(context)

    # Create a CL buffer for each array type parameter
    for info in arg_info:
        if not CLArgType.is_array(info.arg_type):
            buffers.append(None)
        else:
            buffers.append(cl.Buffer(context, cl.mem_flags.WRITE_ONLY if info.is_output else cl.mem_flags.READ_ONLY, info.byte_size))
    
    # Compile the kernel, callable with program.func_name after this step
    program = cl.Program(context, kernel).build()

    def cl_func(*args):
        cl_args = []

        # Check that the output arguments are numpy types
        if any([arg.is_output and not isinstance(arg, np.ndarray) for arg in args]):
            raise Exception("At least one output argument passed is not a numpy array")

        # Prepare arguments
        for i, info in enumerate(arg_info):
            # Replace None by the raw input, used by scalars
            if buffers[i] == None:
                # Convert standard number types to buffer interface
                if not isinstance(args[i], np.ndarray):
                    cl_args.append(np.array(args[i]))
                else:
                    cl_args.append(args[i])
            else:
                cl_args.append(buffers[i])

                # Queue input copying, None means no copying should be done (eg. already copied previously)
                # Also convert the argument to a numpy array if it isn't already one
                if not info.is_output and args[i] is not None:
                    cl.enqueue_copy(queue, buffers[i], args[i] if isinstance(args[i], np.ndarray) else np.array(args[i]))

        # Execute the compiled func
        prog_func = getattr(program, func_name)
        prog_func(queue, dim_shape, None, *cl_args)

        # Copy output
        for i, info in enumerate(arg_info):
            if info.is_output:
                cl.enqueue_copy(queue, args[i], buffers[i])

    return cl_func