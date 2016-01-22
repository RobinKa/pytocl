# Dummy functions only used in OpenCL

def cl_call(name, *args):
    return -1

def cl_inline(value):
    return -1

# Some regularly used dummy functions for convenience

def get_global_id(i):
    return -1

def get_global_size(i):
    return -1

def get_group_id(i):
    return -1

def get_num_groups(i):
    return -1

def get_local_id(i):
    return -1

def get_local_size(i):
    return -1

