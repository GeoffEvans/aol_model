from aol_model.aol_full import AolFull
from aol_model.aod import Aod
from aol_model.ray import Ray
from numpy import array, linspace, exp, power
from aol_model.vector_utils import normalise_list

def set_up_aol( op_wavelength, \
                order=-1, \
                base_freq=40e6, \
                focus_position=[0,0,1e12], \
                focus_velocity=[0,0,0], \
                pair_deflection_ratio=1, \
                ac_power=[1,1,2,2]):
   
    # for centre freq = 40MHz
    orient40  = normalise_list(array([ \
        [ -3.60119805e-02,   1.20407784e-18,  9.99351358e-01], \
        [-0.04799655, -0.03600276,  0.99819844], \
        [-0.01577645, -0.04799655,  0.9987229 ], \
        [ 0.00872665, -0.01570224,  0.99983863]]))
   
    # for centre freq = 35MHz
    orient35 = normalise_list(array([ \
        [ -3.92662040e-02, -0., 9.99228785e-01], \
        [ -0.04613158, -0.03934167,  0.99816036], \
        [ -0.01187817, -0.04380776,  0.99896936], \
        [ 0, -1.17766548e-02,  9.99930653e-01] ]))
    
    # for centre freq = 39MHz     --- rig3
    orient39 = normalise_list(array([ \
        [-0.03634428, 0., 0.99933933], \
        [-0.05410521, -0.03632524,  0.99787429], \
        [-0.022247121509960638, -0.054595032101442259, 0.9982607114648776], \
        [0.010005291640563912, -0.022117399101740584, 0.9997053139781551] ]))

    orient = normalise_list(array([ \
        [-0.036, 0., 1], \
        [-0.054, -0.036,  1], \
        [-0.022, -0.054,  1], \
        [0.0, -0.022, 1] ]))
   
    aod_spacing = array([5e-2] * 3)
    aods = [0]*4
    orientations = orient
    aods[0] = make_aod_wide(orientations[0], [1,0,0])
    aods[1] = make_aod_wide(orientations[1], [0,1,0])
    aods[2] = make_aod_narrow(orientations[2], [-1,0,0])
    aods[3] = make_aod_narrow2(orientations[3], [0,-1,0])

    return AolFull.create_aol(aods, aod_spacing, order, op_wavelength, base_freq, pair_deflection_ratio, focus_position, focus_velocity, ac_power=ac_power)

def get_ray_bundle(op_wavelength, width=15e-3):
    x_array = linspace(-width/2, width/2, 5)
    y_array = x_array
    
    rays = [0] * len(x_array) * len(y_array)
    for xn in range(len(x_array)):
        for yn in range(len(y_array)):    
            rays[xn + yn*len(x_array)] = Ray([x_array[xn],y_array[yn],0], [0,0,1], op_wavelength)
            
    return rays

def p(x, width):
    val = x*0 
    val[x > 0] = exp(-power(x[x > 0], -1) * width)
    return val
    
def q(x, width):
    val = x*0    
    val[[x > 0]] = p(x[x > 0], width) / (p(x[x > 0], width) + p(width - x[x > 0], width))
    return val
    
def r(x, lower, lower_width, upper, upper_width): # 11.13 Priestley, Introduction to Integration
    return q(upper - x, upper_width) * q(x - lower, lower_width)

def transducer_efficiency_narrow(freq): 
    return 1#r(array(freq), 13e6, 10e6, 90e6, 10e6)
def transducer_efficiency_narrow2(freq):
    return transducer_efficiency_narrow(freq)
def transducer_efficiency_wide(freq):
    return 1#r(array(freq), 15e6, 10e6, 85e6, 10e6)
    
def make_aod_wide(orientation, ac_dir):
    return Aod(orientation, ac_dir, 16e-3, 3.3e-3, 8e-3, transducer_efficiency_wide)
def make_aod_narrow(orientation, ac_dir):
    return Aod(orientation, ac_dir, 16e-3, 1.2e-3, 8e-3, transducer_efficiency_narrow)
def make_aod_narrow2(orientation, ac_dir):
    return Aod(orientation, ac_dir, 16e-3, 1.2e-3, 8e-3, transducer_efficiency_narrow2)