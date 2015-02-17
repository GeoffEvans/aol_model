from numpy.linalg import norm
from numpy import dot, outer, zeros, abs, all, isnan, sqrt, logical_not
from scipy.optimize import fsolve
from vector_utils import normalise_list

def original(sum_vector, base_length, normal, multiplier_func):
    
    def zero_func(mismatches):
        wavevector_mismatches = outer(mismatches * 1e6, normal)
        wavevectors_out = sum_vector + wavevector_mismatches
        wavevectors_out_mag1 = norm(wavevectors_out, axis=1)
          
        n_out = multiplier_func(wavevectors_out)
        wavevectors_out_mag2 = n_out * base_length # n_out pretty much constant
        
        return wavevectors_out_mag2 - wavevectors_out_mag1 
    
    initial_guess = zeros(len(sum_vector))
    wavevector_mismatches_mag = fsolve(zero_func, initial_guess, band=(0,0)) * 1e6
    
    wavevector_mismatches = outer(wavevector_mismatches_mag, normal)
    wavevectors_out = sum_vector + wavevector_mismatches

    wavevectors_out_unit = normalise_list(wavevectors_out)
    return (wavevector_mismatches_mag, wavevectors_out_unit, base_length)

def faster_method(sum_vector, base_length, normal, multiplier_func):

    def tail_rec_solve(sv):        
        multiplier = multiplier_func(sv) # ord ref ind
        ratio = base_length * multiplier / norm(sv, axis=1) # desired_wavelength_in_crystal / current_wavelength
        if all(abs(ratio - 1) < 1e-6):
            return sv
        new_sum_vector = sv - outer(dot(sv, normal) * (1 - ratio), normal) # improve approx
        return tail_rec_solve(new_sum_vector)
     
    wavevec_out, valid_wavevecs = precondition(sum_vector, base_length, normal, multiplier_func)
    
    wavevec_out[valid_wavevecs] = tail_rec_solve(wavevec_out[valid_wavevecs])
    wavevectors_out_unit = normalise_list(wavevec_out)
    
    wavevector_mismatches_mag = dot(wavevec_out - sum_vector, normal)    
    
    return (wavevector_mismatches_mag, wavevectors_out_unit, base_length) 
    
def precondition(sum_vectors, base_length, normal, multiplier_func):
    # use of xyz could be slightly misleading: z taken to be aligned with norm
    r_xy = sum_vectors - outer(dot(normal, sum_vectors.transpose()), normal)
    multiplier = multiplier_func(sum_vectors)
    r_z = sqrt( (base_length * multiplier)**2 - norm(r_xy, axis=1)**2 )
    invalid_wavevecs = isnan(r_z) # has gone off the indicatrix
    r_z[invalid_wavevecs] = 1e-6    
    
    valid_wavevecs = logical_not(invalid_wavevecs)
    wavevec_out = r_xy + outer(r_z, normal)
    return wavevec_out, valid_wavevecs
    
if __name__ == "__main__":
    import aol_model.pointing_efficiency as p
    p.plot_fov_lines([1e9], 0)
    
