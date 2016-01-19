import unittest
from pytocl import *

"""Tests for Parameters"""

def one_dim():
    i = get_global_id(0)

def two_dim():
    i = get_global_id(0)
    j = get_global_id(1)

def three_dim():
    i = get_global_id(0)
    j = get_global_id(1)
    k = get_global_id(2)

def array_params(float_array, int_array):
    i = get_global_id(0)

def scalar_params(float, int):
    i = get_global_id(0)

def writable_params(float, int):
    i = get_global_id(0)

class TestParameters(unittest.TestCase):
    def test_one_dim(self):
        kernel = func_to_kernel(CLFuncDesc(one_dim, (1,)))
        expected_header = "kernel void one_dim()"
        self.assertIn(expected_header, kernel)
        expected_one_dim = "int i=get_global_id(0);"
        self.assertIn(expected_one_dim, kernel)

    def test_two_dim(self):
        kernel = func_to_kernel(CLFuncDesc(two_dim, (1,1)))
        expected_header = "kernel void two_dim()"
        self.assertIn(expected_header, kernel)
        expected_one_dim = "int i=get_global_id(0);"
        self.assertIn(expected_one_dim, kernel)
        expected_two_dim = "int j=get_global_id(1);"
        self.assertIn(expected_two_dim, kernel)

    def test_three_dim(self):
        kernel = func_to_kernel(CLFuncDesc(three_dim, (1,1,1)))
        expected_header = "kernel void three_dim()"
        self.assertIn(expected_header, kernel)
        expected_one_dim = "int i=get_global_id(0);"
        self.assertIn(expected_one_dim, kernel)
        expected_two_dim = "int j=get_global_id(1);"
        self.assertIn(expected_two_dim, kernel)
        expected_two_dim = "int k=get_global_id(2);"
        self.assertIn(expected_two_dim, kernel)

    def test_array(self):
        kernel = func_to_kernel(CLFuncDesc(array_params, (1,)).arg(CLArgDesc(CLArgType.float32_array, 100)).arg(CLArgDesc(CLArgType.int32_array, 10)))
        expected_header = "kernel void array_params(const global float* float_array,const global int* int_array)"
        self.assertIn(expected_header, kernel)
        expected_one_dim = "int i=get_global_id(0);"
        self.assertIn(expected_one_dim, kernel)
        
    def test_scalar(self):
        kernel = func_to_kernel(CLFuncDesc(scalar_params, (1,)).arg(CLArgDesc(CLArgType.float32, 100)).arg(CLArgDesc(CLArgType.int32, 10)))
        expected_header = "kernel void scalar_params(const float float,const int int)"
        self.assertIn(expected_header, kernel)
        expected_one_dim = "int i=get_global_id(0);"
        self.assertIn(expected_one_dim, kernel)

    def test_writable(self):
        kernel = func_to_kernel(CLFuncDesc(writable_params, (1,)).arg(CLArgDesc(CLArgType.float32_array, 100), False).arg(CLArgDesc(CLArgType.int32_array, 10), False))
        expected_header = "kernel void writable_params(global float* float,global int* int)"
        self.assertIn(expected_header, kernel)
        expected_one_dim = "int i=get_global_id(0);"
        self.assertIn(expected_one_dim, kernel)

"""Tests for Literals"""

def num_literal():
    i = 5
    f = 3.4
    f_dec_pre = .4
    f_dec_post = 4.

def name_constant_literal():
    b_true = True
    b_false = False
    # TODO: Test None

class TestLiterals(unittest.TestCase):
    def test_num(self):
        kernel = func_to_kernel(CLFuncDesc(num_literal, (1,)))
        expected_int = "i=5"
        self.assertIn(expected_int, kernel)
        expected_float = "f=3.4f"
        self.assertIn(expected_float, kernel)
        expected_float_dec_pre = "f_dec_pre=0.4f"
        self.assertIn(expected_float_dec_pre, kernel)
        expected_float_dec_post = "f_dec_post=4.0f"
        self.assertIn(expected_float_dec_post, kernel)

    def test_name_constant(self):
        kernel = func_to_kernel(CLFuncDesc(name_constant_literal, (1,)))
        expected_true = "b_true=true;"
        self.assertIn(expected_true, kernel)
        expected_false = "b_false=false;"
        self.assertIn(expected_false, kernel)

"""Tests for Comparisons"""

def comparisons():
    a = 3
    b = 4
    b_is_greater = a > b
    b_is_equal = a == b
    b_is_less = a < b
    b_is_not_equal = a != b

class TestComparisons(unittest.TestCase):
    def test_comparisons(self):
        kernel = func_to_kernel(CLFuncDesc(comparisons, (1,)))
        expected_greater = "bool b_is_greater=(a>b);"
        self.assertIn(expected_greater, kernel)
        expected_is_equal = "bool b_is_equal=(a==b);"
        self.assertIn(expected_is_equal, kernel)
        expected_is_less = "bool b_is_less=(a<b);"
        self.assertIn(expected_is_less, kernel)
        expected_is_not_equal = "bool b_is_not_equal=(a!=b);"
        self.assertIn(expected_is_not_equal, kernel)

"""Tests for For loops"""

def for_loop_one_arg():
    for i in range(10):
        pass

def for_loop_two_arg():
    for i in range(10, 20):
        pass

def for_loop_three_arg():
    for i in range(10, 20, 2):
        pass

class TestForLoop(unittest.TestCase):
    def test_one_arg(self):
        kernel = func_to_kernel(CLFuncDesc(for_loop_one_arg, (1,)))
        expected_for = "for(int i=0;i<10;i++)"
        self.assertIn(expected_for, kernel)

    def test_two_arg(self):
        kernel = func_to_kernel(CLFuncDesc(for_loop_two_arg, (1,)))
        expected_for = "for(int i=10;i<20;i++)"
        self.assertIn(expected_for, kernel)

    def test_three_arg(self):
        kernel = func_to_kernel(CLFuncDesc(for_loop_three_arg, (1,)))
        expected_for = "for(int i=10;i<20;i+=2)"
        self.assertIn(expected_for, kernel)

"""Tests for While loops"""

def while_loop(dim):
    while True:
        pass

class TestWhileLoop(unittest.TestCase):
    def test_while_loop(self):
        kernel = func_to_kernel(CLFuncDesc(while_loop, (1,)))
        expected_while = "while(true)"
        self.assertIn(expected_while, kernel)

"""Tests for If statements"""

def if_statement():
    if True:
        pass

def if_else_statement():
    if True:
        pass
    else:
        pass

def if_comparison():
    i = get_global_id(0)
    if i > 4:
        pass

class TestIfStatement(unittest.TestCase):
    def test_if_statement(self):
        kernel = func_to_kernel(CLFuncDesc(if_statement, (1,)))
        expected_if = "if(true)"
        self.assertIn(expected_if, kernel)

    def test_if_else_statement(self):
        kernel = func_to_kernel(CLFuncDesc(if_else_statement, (1,)))
        expected_if = "if(true)"
        self.assertIn(expected_if, kernel)
        expected_else = "else"
        self.assertIn(expected_else, kernel)

    def test_if_comparison(self):
        kernel = func_to_kernel(CLFuncDesc(if_comparison, (1,)))
        expected_if = "if((i>4))"
        self.assertIn(expected_if, kernel)

""" Tests for OpenCL barriers """

def test_local_barrier():
    cl_call("barrier", cl_inline("CLK_LOCAL_MEM_FENCE"))

def test_global_barrier():
    cl_call("barrier", cl_inline("CLK_GLOBAL_MEM_FENCE"))

def test_local_and_global_barrier():
    cl_call("barrier", cl_inline("CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE"))

class TestBarrier(unittest.TestCase):
    def test_local_barrier(self):
        kernel = func_to_kernel(CLFuncDesc(test_local_barrier, (1,)))
        expected = "barrier(CLK_LOCAL_MEM_FENCE);"
        self.assertIn(expected, kernel)

    def test_global_barrier(self):
        kernel = func_to_kernel(CLFuncDesc(test_global_barrier, (1,)))
        expected = "barrier(CLK_GLOBAL_MEM_FENCE);"
        self.assertIn(expected, kernel)

    def test_local_and_global_barrier(self):
        kernel = func_to_kernel(CLFuncDesc(test_local_and_global_barrier, (1,)))
        expected = "barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);"
        self.assertIn(expected, kernel)

""" Tests for miscalleneous things """

class SomeClass:
    def class_func():
        pass

def return_func():
    return

class TestMisc(unittest.TestCase):
    def test_class_func(self):
        kernel = func_to_kernel(CLFuncDesc(SomeClass.class_func, (1,)))

    def test_return_func(self):
        kernel = func_to_kernel(CLFuncDesc(return_func, (1,)))
        expected_return = "return;"
        self.assertIn(expected_return, kernel)

if __name__ == "__main__":
    unittest.main()
