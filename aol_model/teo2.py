from scipy.interpolate import splrep, splev 
from scipy.constants import h, c, e, pi
import optically_uniaxial as oua
from numpy import power, arange, array, sqrt
import numpy as np
from memoize import memoized

accuracy = 7

def calc_refractive_indices(angles, wavelength_vac):
    wavelength_vac_rounded = np.round(wavelength_vac, accuracy) # to nearest 10nm
    (n_e, n_o) = ref_ind_lookup(wavelength_vac_rounded)
    return (splev(abs(angles),n_e),splev(abs(angles),n_o))       

@memoized
def ref_ind_lookup(wavelength_vac_rounded):
    angles_stored = arange(0, pi/2+1e-4, 1e-4)
    n_fixed_wavelength = oua.calc_refractive_indices(angles_stored, \
        get_relative_impermeability_eigenvals(wavelength_vac_rounded), get_activity_vector(wavelength_vac_rounded))
    return (splrep(angles_stored, n_fixed_wavelength[0]), splrep(angles_stored, n_fixed_wavelength[1]))

# See Uchida 1971 for constants and formulas           
def get_ref_ind(wavelength_vac):
    F1 = array([220.6, 241.0])
    F2 = array([25.55, 34.20])
    E1_F = array([9.24, 9.24])
    E2_F = array([4.70, 4.71])
    E = h*c/e/wavelength_vac
    n_sqr = 1 + F1 / (power(E1_F, 2.) - power(E, 2.)) \
              + F2 / (power(E2_F, 2.) - power(E, 2.)) # Uchida 1971 (4)
    return sqrt(n_sqr)
    
def get_relative_impermeability_eigenvals(wavelength_vac):
    n = get_ref_ind(wavelength_vac)
    return power([n[0], n[0], n[1]], -2.)

def get_activity_vector(wavelength_vac):
    G1 = 0.8838
    G2 = 0.08754
    E1_G = 9.31
    E2_G = 4.69
    E = h*c/e/wavelength_vac 
    rotary_power_rad_per_micrometer = G1 * power(E, 2.) * power((power(E1_G, 2.) - power(E, 2.)), -2.) \
                                    + G2 * power(E, 2.) * power((power(E2_G, 2.) - power(E, 2.)), -2.) # Uchida 1971 (7)
    
    # Following two lines useful for overriding model above
    #rotary_power_deg_per_mm = 58.5
    #rotary_power_rad_per_micrometer = rotary_power_deg_per_mm / 1000 * pi / 180 
    
    return rotary_power_rad_per_micrometer * 1e6 * wavelength_vac / pi / power(get_ref_ind(wavelength_vac)[0], 3.) # Xu & St (1.78)
      
def plot_refractive_index(wavelength): 
    from pylab import plot,xlabel,ylabel,title,grid,show
    
    angles = arange(-pi/2, pi/2, pi/360)
    n = calc_refractive_indices(angles, wavelength)
    
    plot(angles, n[0])
    plot(angles, n[1])
    
    xlabel('angle / rad')
    ylabel('refractive index')
    title('Ordinary under Extraordinary')
    grid(True)
    show()

if __name__ == '__main__':
    plot_refractive_index(800e-9)