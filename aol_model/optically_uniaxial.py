from numpy import array, diag, dot, atleast_1d, sqrt, power

# Follow the calculation in Xu&St Section 1.3
# z axis is taken as direction of the optical wavevector

def calc_refractive_indices(angles, relative_impermeability_eigenvals, activity_vector):
    (sqrt_term,eigensum,_) = get_imperm_properties(angles, relative_impermeability_eigenvals, activity_vector)
    
    ext_recip_sqr = 0.5 * ( eigensum - sqrt_term ) # Xu&St (1.62)
    ord_recip_sqr = 0.5 * ( eigensum + sqrt_term )

    n_e = power(ext_recip_sqr, -0.5)
    n_o = power(ord_recip_sqr, -0.5)
    
    return (n_e, n_o)

def calc_polarisations(angles, relative_impermeability_eigenvals, activity_vector):
    (sqrt_term,_,eigendiff) = get_imperm_properties(angles, relative_impermeability_eigenvals, activity_vector)
    
    p_e =  0.5j / activity_vector * ( sqrt_term - eigendiff ) # Xu&St (1.62)
    p_o = -0.5j / activity_vector * ( sqrt_term + eigendiff )
    
    return (p_e, p_o)

def get_imperm_properties(angles, relative_impermeability_eigenvals, activity_vector):
    angles = atleast_1d(angles)
    transverse_imperm_eigvals = find_transverse_imperm_eigvals(angles, relative_impermeability_eigenvals)
    eigenval1 = transverse_imperm_eigvals[:,0]
    eigenval2 = transverse_imperm_eigvals[:,1]     
    
    sqrt_term = sqrt( power(eigenval1 - eigenval2, 2.) + 4 * activity_vector**2)
    eigensum = eigenval1 + eigenval2
    eigendiff = eigenval2 - eigenval1
    
    return (sqrt_term, eigensum, eigendiff)

def find_transverse_imperm_eigvals(angles, relative_impermeability_eigenvals):
    principal_imperm = diag(relative_impermeability_eigenvals)    
    rotation_matrix = get_yz_rotation_matrix(angles)
    relative_impermeability_matrix = array([dot(dot(rot,principal_imperm),rot.T) for rot in rotation_matrix]) # Xu&St (1.59) 
    return array([[x[0,0],x[1,1]] for x in relative_impermeability_matrix]) # eigenvals for transverse components
   
def get_yz_rotation_matrix(angles):
    from numpy import cos, sin
    return array([[ [1,     0,       0  ], \
                    [0,     cos(a), -sin(a)], \
                    [0,     sin(a),  cos(a)]] for a in angles])