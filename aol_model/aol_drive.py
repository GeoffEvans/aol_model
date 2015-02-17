from numpy import array, append, dtype

def calculate_drive_freq_4(order, op_wavelength, ac_velocity, aod_spacing, crystal_thickness, base_freq, pair_deflection_ratio, focus_position, focus_velocity):
    
    spacing = get_reduced_spacings(crystal_thickness, append(aod_spacing, focus_position[2]))
    xy_deflection = array(focus_position[0:2])
    xy_focus_velocity = array(focus_velocity[0:2])
    
    const = find_constant(order, op_wavelength, ac_velocity, spacing, base_freq, pair_deflection_ratio, xy_deflection)
    linear = find_linear(order, op_wavelength, ac_velocity, spacing, base_freq, pair_deflection_ratio, xy_focus_velocity)
    quadratic = [0]*4 # not yet implemented
    return (const, linear, quadratic)

def get_reduced_spacings(crystal_thickness, spacing):
    spacing_correction = array(crystal_thickness) * (1 - 1/2.26) # 2.26 is approx TeO2 refractive index
    spacing = array(spacing, dtype=dtype(float))
    return spacing - spacing_correction 
    
def find_constant(order, op_wavelength, ac_velocity, spacing, base_freq, pair_deflection_ratio, xy_deflection):
    # for constant components, choose r to represent the ratio of ANGULAR deflection on the first of the pair to the second of the pair
    # traditionally, this means r = 1, while all on the second would be r = 0    

    multiplier = ac_velocity / (op_wavelength * order)
    dfx = multiplier * xy_deflection[0] / (pair_deflection_ratio * spacing[0:4].sum() + spacing[2:4].sum()) # new neat code
    dfy = multiplier * xy_deflection[1] / (pair_deflection_ratio * spacing[1:4].sum() + spacing[3:4].sum())
    
    return array([base_freq + pair_deflection_ratio * dfx, \
                  base_freq + pair_deflection_ratio * dfy, \
                  base_freq - dfx, \
                  base_freq - dfy ])

def find_linear(order, op_wavelength, ac_velocity, spacing, base_freq, pair_deflection_ratio, xy_focus_velocity):
    
    v = xy_focus_velocity / ac_velocity
    
    factors = array([(1 + v[0])/((1 + v[0]) * spacing[0:2].sum() + 2 * spacing[2:4].sum()), \
                  (1 + v[1])/((1 + v[1]) * spacing[1:3].sum() + 2 * spacing[3:4].sum()), \
                  (1 - v[0])/(2 * spacing[2:4].sum()), \
                  (1 - v[1])/(2 * spacing[3:4].sum()) ])

    return (ac_velocity**2 / op_wavelength) * factors / order 


if __name__ == '__main__':
    print calculate_drive_freq_4(-1, 800e-9, 613, [5e-2]*3, 0e-3, 40e6, 0, [0,0.01,1], [0,0,0])