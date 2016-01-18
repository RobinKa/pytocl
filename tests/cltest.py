import unittest
import numpy as np
from pytocl import CLArgType, CLArgDesc, CLFuncDesc, CLFunc

def add_func(dim, a, b, output):
    output[dim] = a[dim] + b[dim]

def add_inplace_func(dim, a, b):
    a[dim] = a[dim] + b[dim]

class SingleFuncTest(unittest.TestCase):
    def test_add_float(self):
        shape = (100,)

        desc_a = CLArgDesc(CLArgType.float32_array, array_size=100)
        desc_b = CLArgDesc(CLArgType.float32_array, array_size=100)
        desc_c = CLArgDesc(CLArgType.float32_array, array_size=100)

        desc_add_func = (CLFuncDesc(add_func, shape)
                        .arg(desc_a).copy_in()
                        .arg(desc_b).copy_in()
                        .arg(desc_c, False).copy_out())

        cl_add = CLFunc(desc_add_func).compile()

        a = 2.0 * np.ones(shape, dtype=np.float32)
        b = 3.0 * np.ones(shape, dtype=np.float32)
        c = np.zeros(shape, dtype=np.float32)
        cl_add({ desc_a: a, desc_b: b, desc_c: c })
        
        self.assertTrue(all([x == 5.0 for x in c]))

    def test_add_int(self):
        shape = (100,)

        desc_a = CLArgDesc(CLArgType.int32_array, array_size=100)
        desc_b = CLArgDesc(CLArgType.int32_array, array_size=100)
        desc_c = CLArgDesc(CLArgType.int32_array, array_size=100)

        desc_add_func = (CLFuncDesc(add_func, shape)
                        .arg(desc_a).copy_in()
                        .arg(desc_b).copy_in()
                        .arg(desc_c, False).copy_out())

        cl_add = CLFunc(desc_add_func).compile()

        a = 2 * np.ones(shape, dtype=np.int32)
        b = 3 * np.ones(shape, dtype=np.int32)
        c = np.zeros(shape, dtype=np.int32)
        cl_add({ desc_a: a, desc_b: b, desc_c: c })
        
        self.assertTrue(all([x == 5 for x in c]))

    def test_add_inplace(self):
        shape = (100,)

        desc_a = CLArgDesc(CLArgType.float32_array, array_size=100)
        desc_b = CLArgDesc(CLArgType.float32_array, array_size=100)

        desc_add_inplace_func = (CLFuncDesc(add_inplace_func, shape)
                        .arg(desc_a, False).copy_in().copy_out()
                        .arg(desc_b).copy_in())

        cl_add_inplace = CLFunc(desc_add_inplace_func).compile()

        a = 2.0 * np.ones(shape, dtype=np.float32)
        b = 3.0 * np.ones(shape, dtype=np.float32)
        cl_add_inplace({ desc_a: a, desc_b: b })
        
        self.assertTrue(all([x == 5.0 for x in a]))

class SequentialFuncTest(unittest.TestCase):
    def test_aux(self):
        shape = (100,)

        # Computes
        # 1. C = A + B
        # 2. D = B + C
        # A: Copied input
        # B: Copied input
        # C: Auxiliary variable A+B
        # D: Copied output B+C

        desc_a = CLArgDesc(CLArgType.float32_array, array_size=100)
        desc_b = CLArgDesc(CLArgType.float32_array, array_size=100)
        desc_c = CLArgDesc(CLArgType.float32_array, array_size=100)
        desc_d = CLArgDesc(CLArgType.float32_array, array_size=100)

        # C = A + B
        desc_func_f = (CLFuncDesc(add_func, shape)
                        .arg(desc_a).copy_in()
                        .arg(desc_b).copy_in()
                        .arg(desc_c, False))

        # D = B + C
        desc_func_g = (CLFuncDesc(add_func, shape)
                        .arg(desc_b)
                        .arg(desc_c)
                        .arg(desc_d, False).copy_out())

        cl_add = CLFunc(desc_func_f, desc_func_g).compile()

        a = 2.0 * np.ones(shape, dtype=np.float32)
        b = 3.0 * np.ones(shape, dtype=np.float32)
        d = np.zeros(shape, dtype=np.float32)

        cl_add({ desc_a: a, desc_b: b, desc_d: d })
        
        self.assertTrue(all([x == 8.0 for x in d]))

if __name__ == "__main__":
    unittest.main()