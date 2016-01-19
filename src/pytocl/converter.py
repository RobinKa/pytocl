import ast
import inspect
from copy import copy
import pyopencl as cl
from .descriptors import CLArgType

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

    def __init__(self, func_desc):
        """Initializes a new CLVisitor

        Keyword arguments:
        func_desc -- CLFuncDesc describing the function to convert
        """
        super().__init__()

        if len(func_desc.global_size) <= 0 or len(func_desc.global_size) > 3:
            raise Exception("Unsupported dimension for global work size " + str(len(dim_shape)))
        
        if func_desc.local_size != None and len(func_desc.local_size) != len(func_desc.global_size):
            raise Exception("Dimensions of local work size have to match dimensions of global work size")

        self.func_desc = func_desc
        
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
        if node.name != self.func_desc.func_name:
            raise Exception("Node function name is not identical with passed name")
        
        func_args = [arg.arg for arg in node.args.args]

        self.write("kernel void " + self.func_desc.func_name + "(")

        # Write arguments
        # TODO: Check whether args are identical to tree args
        arg_decls = []
        for arg, arg_desc in zip(func_args, self.func_desc.arg_descs):
            decl = ""
            if self.func_desc.is_readonly(arg_desc):
                decl += "const "

            if self.func_desc.is_local(arg_desc):
                decl += "local "
            # Make all non-local arrays global by default
            elif CLArgType.is_array(arg_desc.arg_type):
                decl += "global "

            decl += CLArgType.get_cl_type_name(arg_desc.arg_type)
            decl += " " + arg
            arg_decls.append(decl)

        self.append(",".join(arg_decls))
        self.append(")")
        self.write("{")
        self.push_block()

        # Declare arguments
        for arg in func_args:
            self.declare_var(arg)

        self.generic_visit(node)

        self.pop_block()

        self.write("}")

    def visit_Name(self, node):
        if not self.is_var_declared(node.id):
            if not isinstance(node.ctx, ast.Store):
                raise Exception("Tried to load or delete a variable which was not yet declared: " + node.id)

            self.declare_var(node.id)

            # TODO: Proper type inference
            type_name = "float"
            if node.id.startswith("i_") or node.id in ["i", "j", "k"]:
                type_name = "int"
            elif node.id.startswith("b_"):
                type_name = "bool"

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

    def visit_NameConstant(self, node):
        value = node.value

        if value == True:
            self.append("true")
        elif value == False:
            self.append("false")
        elif value == None:
            self.append("NULL")
        else:
            raise Exception("Unknown Name Constant")

    def visit_String(self, node):
        self.append(node.s)

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
            raise Exception("Slice subscripts are not supported")

        if isinstance(slice.value, ast.Tuple):
            raise Exception("Only single dimensional subscripts are supported")

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

        if len(orelse) != 0:
            raise NotImplementedError("For loop orelse not yet implemented")

        # Check that the iter function is range()
        if not isinstance(iter, ast.Call) or not isinstance(iter.func, ast.Name) or not iter.func.id == "range":
            raise Exception("Unsupported for loop:", ast.dump(node))

        # Handle the different arguments of range()
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

    def visit_While(self, node):
        test = node.test
        body = node.body
        orelse = node.orelse # TODO

        if len(orelse) != 0:
            raise NotImplementedError("For loop orelse not yet implemented")

        self.write("while(")
        self.visit(test)
        self.append(")")

        self.write("{")

        self.push_block()

        for b in body:
            self.visit(b)

        self.pop_block()

        self.write("}")

    def visit_Break(self, node):
        self.write("break;")

    def visit_Continue(self, node):
        self.write("continue;")

    def visit_IfExp(self, node):
        test = node.test
        body = node.body
        orelse = node.orelse

        self.visit(test)
        self.append("?")
        self.visit(body)
        self.append(":")
        self.visit(orelse)

    def visit_Call(self, node):
        func = node.func
        func_name = func.id
        args = node.args
        keyword = node.keywords
        
        if not isinstance(func, ast.Name):
            raise Exception("Unsupported function call")

        # Handle open cl calls, set func name and trim first argument
        if func.id == "cl_call":
            func_name = args[0].s
            args = copy(args[1:]) # Copy because else we're modifying the AST
        
        # Handle open cl flags
        if func.id == "cl_inline":
            self.append(args[0].s)
        # Handle normal function calls
        else:
            # TODO: Check if function is supported
            self.append(func_name)
            self.append("(")
            for i, arg in enumerate(args):
                self.visit(arg)
                if i != len(args)-1:
                    self.append(",")
            self.append(")")

    def visit_Return(self, node):
        value = node.value

        if value != None:
            raise Exception("Only empty returns are supported")

        self.write("return;")

def func_to_kernel(func_desc):
    """Converts a python function to an OpenCL kernel as a string

    Keyword arguments:
    func_desc -- the descriptor for the function to convert
    """

    source = inspect.getsource(func_desc.func)

    # Remove the unused indents (when a function isnt declared at root)
    unused_indents = source.split("def", 1)[0]

    # Make sure all characters are tabs or spaces
    assert(all([c == " " or c == "\t" for c in unused_indents]))

    source = source[len(unused_indents):].replace("\n" + unused_indents, "\n")

    tree = ast.parse(source)

    visitor = CLVisitor(func_desc)
    visitor.visit(tree)

    return visitor.get_code()