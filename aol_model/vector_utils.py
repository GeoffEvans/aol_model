from numpy import dot, outer, array, arccos, sum
from numpy.linalg import norm

def normalise(vector):
    return vector/norm(vector, axis=0)

def normalise_list(vectors):
    mags = norm(vectors, axis=1)
    mags[mags == 0] = 1 # avoid divide by zero error: [0,0,0] normalises to [0,0,0]
    return (array(vectors).T/mags).T

def perpendicular_component(vector, unit_normal):
    return vector - dot(vector, unit_normal) * unit_normal

def perpendicular_component_list(vector_list, unit_normal):
    return vector_list - outer( dot(vector_list, unit_normal), unit_normal )

def angle_between_unit_vectors(v1, v2):
    dot_prods = dot(v1, v2)
    return arccos(dot_prods)
    
def angle_between_unit_vectors_list(v1, v2):
    dot_prods = sum(v1 * v2, axis=1)
    return arccos(dot_prods)