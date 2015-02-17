from numpy import array, dot

def check_is_unit_vector(vector):
    if abs(dot(vector,vector) - 1) > 1e-14:
        raise ValueError("vector must be unit length")
    
def check_is_of_length(desired, arr):
    length = arr.shape[0]
    if not length == desired:
        raise ValueError("array has wrong size")
    
def check_is_val(var,val):
    if not var == val:
        raise ValueError("variable has wrong value")
    
def check_is_singleton(var):
    if not array(var).size == 1:
        raise ValueError("variable is a list")