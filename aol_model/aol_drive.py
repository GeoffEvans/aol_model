"""The aol_drive module handles the calculation of drive parameters for an AOL."""

from numpy import array, append, dtype, sqrt

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

def calculate_drive_freq_6(order, op_wavelength, ac_velocity, aod_spacing, crystal_thickness, base_freq, pair_deflection_ratio, focus_position, focus_velocity):
    quadratic = array( [0]*6 )

    spacing = get_reduced_spacings(crystal_thickness, append(aod_spacing, focus_position[2]))
    shift = - ac_velocity / op_wavelength * array(focus_position[0:2]) # (-) for -1 mode
    const = base_freq + array([0, 0, 0, -shift[1] / sqrt(3) / spacing[3:6].sum(), shift[0] / spacing[4:6].sum(), shift[1] / sqrt(3) / spacing[5]])
    #const = base_freq + array([0, 0, 0, shift[0] / spacing[3:6].sum(), shift[1] / sqrt(3) / spacing[4:6].sum(), shift[1] / sqrt(3) / spacing[5]])

    z = spacing[-1];
    v = focus_velocity[0] / ac_velocity
    fac = 100000*sqrt(160000000000000000000000000000000000000000*z**8 - 14572800000000000000000000000000000000000*z**7*v + 99580800000000000000000000000000000000000*z**7 + 331822656000000000000000000000000000000*z**6*v**2 - 7576617312000000000000000000000000000000*z**6*v + 26536595184000000000000000000000000000000*z**6 + 135686017043280000000000000000000000000*z**5*v**2 - 164355328610352000000000000000000000000*z**5*v + 3951999537393600000000000000000000000000*z**5 + 200714563865692800000000000000000000*z**4*v**3 + 22052317975196176800000000000000000000*z**4*v**2 - 192563322410924942400000000000000000000*z**4*v + 359487217756191571200000000000000000000*z**4 + 54788839447099736088000000000000000*z**3*v**3 + 1810806508915961684688000000000000000*z**3*v**2 - 13129758774547585039584000000000000000*z**3*v + 20428755008473152779328000000000000000*z**3 + 18737905508871124612410000000000*z**2*v**4 + 5450820578828526927195720000000000*z**2*v**3 + 78301044153351213892295400000000000*z**2*v**2 - 519401755515190682849172480000000000*z**2*v + 707123870471722746706202880000000000*z**2 + 2818555746644394564198712200000*z*v**4 + 231196795398228695070011130000000*z*v**3 + 1660564224525949743596091328800000*z*v**2 - 11001839737339016029887428174400000*z*v + 13605832459795647768150598214400000*z + 105991788852562457586692572281*v**4 + 3443384562216748937737147375848*v**3 + 13218597203822813286369339129720*v**2 - 96087210765642630292884913412064*v + 111255268037351854413413765243664)
    linear = - (ac_velocity**2 / op_wavelength) * array([ \
        (549973834719289200000*v - 80802964519200000000000*z + fac + 28727027970300000000000*z*v + 1141210700190000000000*z*v**2 + 421691292000000000000000*z**2*v + 1821600000000000000000000*z**3*v - 1631461392000000000000000*z**2 - 13662000000000000000000000*z**3 - 40000000000000000000000000*z**4 + 84527115437486700000*v**2 - 1387081003857555600000)/(18216*(11385000000000000000*z**3*v - 200000000000000000000*z**4 - 68689500000000000000*z**3 + 2644211790000000000*z**2*v - 8347413690000000000*z**2 + 5902813966500000*z*v**2 + 179511131403450000*z*v - 419187240717300000*z + 439063108456203*v**2 + 3436838663017716*v - 7249353394646484)), \
        (836708925955993200000*v - 79281350252280000000000*z + fac + 44126813740680000000000*z*v + 511577210430000000000*z*v**2 + 712036116000000000000000*z**2*v + 3643200000000000000000000*z**3*v - 1615331124000000000000000*z**2 - 13662000000000000000000000*z**3 - 40000000000000000000000000*z**4 + 39127392658341900000*v**2 - 1354159567456304400000)/(36432*(6831000000000000000*z**3*v - 100000000000000000000*z**4 - 31878000000000000000*z**3 + 1313464680000000000*z**2*v - 3593305237500000000*z**2 + 590281396650000*z*v**2 + 82393444949062500*z*v - 172263787589025000*z + 44802358005735*v**2 + 1654700422345146*v - 2962929276112608)), \
        (348462784489050000000*v - 58450975632720000000000*z + fac + 20961548263260000000000*z*v + 432873024210000000000*z*v**2 + 352561572000000000000000*z**2*v + 1821600000000000000000000*z**3*v - 1304247384000000000000000*z**2 - 12144000000000000000000000*z**3 - 40000000000000000000000000*z**4 + 32556380150834100000*v**2 - 895117926022729200000)/(36432*(2500000000*z**2 + 303600000*z + 9793377)*(1821600000*z - 57032019*v - 683100000*z*v + 10000000000*z**2 + 86412150)), \
        (77549858155440000000000*z - 674225707588527600000*v - fac - 32557298366340000000000*z*v + 432873024210000000000*z*v**2 - 449343180000000000000000*z**2*v - 1821600000000000000000000*z**3*v + 1613026800000000000000000*z**2 + 13662000000000000000000000*z**3 + 40000000000000000000000000*z**4 + 32556380150834100000*v**2 + 1252209312794365200000)/(291456*(625000000*z**2 + 71156250*z + 1728243)*(3529350000*z - 84683907*v - 1138500000*z*v + 20000000000*z**2 + 146324574)), \
        (85262868405000000000000*z + 169253352466110000000*v - fac + 5876579237760000000000*z*v + 275464651770000000000*z*v**2 + 48390804000000000000000*z**2*v + 1679852196000000000000000*z**2 + 13662000000000000000000000*z**3 + 40000000000000000000000000*z**4 + 21206449456047900000*v**2 + 1462747208489463600000)/(4554*(800000000000000000000*z**4 + 224664000000000000000*z**3 + 483908040000000000*z**2*v + 20819567340000000000*z**2 + 55224103997700000*z*v + 770688881285400000*z + 26881414803441*v**2 + 1358009251551612*v + 9898998034037508)), \
        1/(3*z)])

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
        #min_pdr_x = (multiplier * xy_deflection[0] / (base_freq - 30e6) - spacing[2:4].sum()) / spacing[0:4].sum() # for LFL of 30 MHz
        #min_pdr_y = (multiplier * xy_deflection[1] / (base_freq - 30e6) - spacing[3:4].sum()) / spacing[1:4].sum()

        # for linear LFL
        z_x = spacing[2:4].sum()
        L_x = spacing[0:2].sum()
        x_on_z = ac_velocity / op_wavelength * xy_deflection[0] / z_x
        if x_on_z > -18e6 * (1 + L_x/z_x):
            min_pdr_x = - (18e6 + 2.15 * x_on_z) / (x_on_z + 18e6 * (L_x/z_x + 1))
        else:
            min_pdr_x = 10

        z_y = spacing[3:4].sum()
        L_y = spacing[1:3].sum()
        y_on_z = ac_velocity / op_wavelength * xy_deflection[1] / z_y
        if y_on_z > -18e6 * (1 + L_y/z_y):
            min_pdr_y = - (18e6 + 2.15 * y_on_z) / (y_on_z + 18e6 * (L_y/z_y + 1))
        else:
            min_pdr_y = 10

        pair_deflection_ratio_x = max(0, min_pdr_x)
        pair_deflection_ratio_y = max(0, min_pdr_y)
    else:
        pair_deflection_ratio_x = pair_deflection_ratio
        pair_deflection_ratio_y = pair_deflection_ratio

    dfx = multiplier * xy_deflection[0] / (pair_deflection_ratio_x * spacing[0:4].sum() + spacing[2:4].sum())
    dfy = multiplier * xy_deflection[1] / (pair_deflection_ratio_y * spacing[1:4].sum() + spacing[3:4].sum())

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