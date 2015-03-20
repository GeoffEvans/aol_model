"""The aol_drive module handles the calculation of drive parameters for an AOL."""

from numpy import array, append, dtype

def calculate_drive_freq_4(order, op_wavelength, ac_velocity, aod_spacing, crystal_thickness, base_freq, pair_deflection_ratio, focus_position, focus_velocity):
    """The top level function for calculating drive frequencies. The centre
    frequencies are calculated first (const) before linear chirps.
    The quadratic chirps have not been implemented."""
    spacing = get_reduced_spacings(crystal_thickness, append(aod_spacing, focus_position[2]))
    xy_deflection = array(focus_position[0:2])
    xy_focus_velocity = array(focus_velocity[0:2])

    const = find_constant(order, op_wavelength, ac_velocity, spacing, base_freq, pair_deflection_ratio, xy_deflection)
    linear = find_linear(order, op_wavelength, ac_velocity, spacing, base_freq, pair_deflection_ratio, xy_focus_velocity)
    quadratic = [0]*4 # requires a value to be returned
    return (const, linear, quadratic)

def get_reduced_spacings(crystal_thickness, spacing):
    """Account for the thickness of the crystals. The higher refractive index causes
    a lower curvature. The adjustments are made to ensure the correct axial focal position
    but the orientation of the crystals away from the z-axis will lead to some
    displacement from the desired lateral focal position. """
    spacing_correction = array(crystal_thickness) * (1 - 1/2.26) # 2.26 is approx TeO2 refractive index
    spacing = array(spacing, dtype=dtype(float))
    return spacing - spacing_correction

def find_constant(order, op_wavelength, ac_velocity, spacing, base_freq, pair_deflection_ratio, xy_deflection):
    """Calculate the centre frequency for each of the four AODs."""
    # for constant components, choose r to represent the ratio of ANGULAR deflection on the first of the pair to the second of the pair
    # traditionally, this means r = 1, while all on the second would be r = 0
    multiplier = ac_velocity / (op_wavelength * order)
    
    if pair_deflection_ratio is None:
        min_pdr_x = (multiplier * xy_deflection[0] / (base_freq - 30e6) - spacing[2:4].sum()) / spacing[0:4].sum()
        min_pdr_y = (multiplier * xy_deflection[1] / (base_freq - 30e6) - spacing[3:4].sum()) / spacing[1:4].sum()    
        pair_deflection_ratio_x = max(0, min_pdr_x)
        pair_deflection_ratio_y = max(0, min_pdr_y)
    else:
        pair_deflection_ratio_x = pair_deflection_ratio
        pair_deflection_ratio_y = pair_deflection_ratio
        
    dfx = multiplier * xy_deflection[0] / (pair_deflection_ratio_x * spacing[0:4].sum() + spacing[2:4].sum())
    dfy = multiplier * xy_deflection[1] / (pair_deflection_ratio_y * spacing[1:4].sum() + spacing[3:4].sum())

    dfx = dfx if base_freq - 30e6 > dfx else base_freq - 30e6
    dfy = dfy if base_freq - 30e6 > dfy else base_freq - 30e6
    
    return array([base_freq + pair_deflection_ratio_x * dfx, \
                  base_freq + pair_deflection_ratio_y * dfy, \
                  base_freq - dfx, \
                  base_freq - dfy ])

def find_linear(order, op_wavelength, ac_velocity, spacing, base_freq, pair_deflection_ratio, xy_focus_velocity):
    """Calculate the linear chirps for each of the four AODs."""
    v = xy_focus_velocity / ac_velocity

    factors = array([(1 + v[0])/((1 + v[0]) * spacing[0:2].sum() + 2 * spacing[2:4].sum()), \
                  (1 + v[1])/((1 + v[1]) * spacing[1:3].sum() + 2 * spacing[3:4].sum()), \
                  (1 - v[0])/(2 * spacing[2:4].sum()), \
                  (1 - v[1])/(2 * spacing[3:4].sum()) ])

    return (ac_velocity**2 / op_wavelength) * factors / order


if __name__ == '__main__':
    print calculate_drive_freq_4(-1, 800e-9, 613, [5e-2]*3, 0e-3, 40e6, 0, [0,0.01,1], [0,0,0])