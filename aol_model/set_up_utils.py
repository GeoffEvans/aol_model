"""Helper functions to create rays, AODs, and AOLs."""

from aol_model.aol_full import AolFull
from aol_model.aod import Aod
from aol_model.ray import Ray
from numpy import array, linspace, exp, power, meshgrid, cos, sin, sqrt, cumsum, tile
from scipy.constants import pi
from aol_model.vector_utils import normalise_list

def set_up_aol( op_wavelength, \
                order=-1, \
                base_freq=39e6, \
                focus_position=[0,0,1e12], \
                focus_velocity=[0,0,0], \
                pair_deflection_ratio=1, \
                ac_power=[1.5,1.5,2,2]):
    """Create an AolFull instance complete with Aods. """
    orient_39_920 = normalise_list(array([ \
        [-0.0355, 0., 1], \
        [-0.0585, -0.0365,  1], \
        [-0.0255, -0.0585,  1], \
        [0.0, -0.026, 1] ])) # 0.022

    aod_spacing = array([5e-2] * 3)
    aods = [0]*4
    orientations = orient_39_920
    aods[0] = make_aod_wide(orientations[0], [1,0,0])
    aods[1] = make_aod_wide(orientations[1], [0,1,0])
    aods[2] = make_aod_narrow(orientations[2], [-1,0,0])
    aods[3] = make_aod_narrow(orientations[3], [0,-1,0])

    return AolFull.create_aol(aods, aod_spacing, order, op_wavelength, base_freq, pair_deflection_ratio, focus_position, focus_velocity, ac_power=ac_power)

def set_up_aol6( op_wavelength, \
                order=-1, \
                base_freq=39e6, \
                focus_position=[0,0,1e12], \
                focus_velocity=[0,0,0], \
                pair_deflection_ratio=0, \
                ac_power=[1.5,1.5,1.5,2,2,2]):
    """Create a 6 AOD AolFull instance complete with Aods. """

    dd = array([[0.5,-0.5*sqrt(3),0], [1,0,0], [0.5,0.5*sqrt(3),0], [-0.5,0.5*sqrt(3),0], [-1,0,0], [-0.5,-0.5*sqrt(3),0]])
    orient_39_920 = normalise_list(-0.0365 * dd - 0.0585 * cumsum(dd, axis=0)[[5,0,1,2,3,4],:] + tile([0,0,1], (6,1)))

    aod_spacing = array([5e-2] * 5)
    orientations = orient_39_920
    aods = [0]*6
    aods[0] = make_aod_wide(orientations[0], dd[0])
    aods[1] = make_aod_wide(orientations[1], dd[1])
    aods[2] = make_aod_wide(orientations[2], dd[2])
    aods[3] = make_aod_narrow(orientations[3], dd[3])
    aods[4] = make_aod_narrow(orientations[4], dd[4])
    aods[5] = make_aod_narrow(orientations[5], dd[5])

    return AolFull.create_aol(aods, aod_spacing, order, op_wavelength, base_freq, pair_deflection_ratio, focus_position, focus_velocity, ac_power=ac_power)

def get_ray_bundle(op_wavelength, width=15e-3):
    """Create a grid of rays. Useful for passing into an Aol instance."""
    r_array = width / 4 * linspace(-1, 1, 5)
    angle_array = linspace(0, pi, 5)[:-1]
    r_mesh, angle_mesh = meshgrid(r_array, angle_array)

    rays = []
    for r, ang in zip(r_mesh.ravel(), angle_mesh.ravel()):
            x = r * cos(ang)
            y = r * sin(ang)
            rays.append(Ray([x,y,0], [0,0,1], op_wavelength))
    return rays

def get_single_ray(op_wavelength, width=15e-3):
    return [Ray([0,0,0], [0,0,1], op_wavelength)]

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

# very ugly code used here to store some data to use in next two functions
import scipy.interpolate as interp
freq_points = [20,22,23,24,25,27,28,30,33,35,37,38,39,40,41,42,43,45,47,50]
narrow_profile_points = [0.54180401708669412, 0.56991991404193254, 0.64237496788483506, 0.71376549924079602, 0.65696105269933802, 0.4909423776444925, 0.4420660934868994, 0.50424863916826046, 0.68557630125400504, 0.80270163439279862, 0.84811765360796099, 0.92148223971098142, 0.95345973740663914, 0.93341096518369848, 0.87226301186133615, 0.84951540465223863, 0.8275046815345859, 0.77382549757853791, 0.7132103559071139, 0.67349178617290195]
narrow_acc_profile = interp.splrep(freq_points, narrow_profile_points)
wide_profile_points = [0.086546704758180285, 0.22535561949294833, 0.18450848585508078, 0.19705430868588988, 0.19059445638313324, 0.2473328903903686, 0.34369865160925528, 0.45973429646282554, 0.60185130925914399, 0.59476921046284681, 0.61774875827843778, 0.62426321922297623, 0.60627059841361375, 0.61539915555987479, 0.59678876895832855, 0.61896602572507742, 0.63529516603260705, 0.59356746666510929, 0.52713268254659917, 0.37368409897364563]
wide_acc_profile = interp.splrep(freq_points, wide_profile_points)

def transducer_efficiency_narrow(freq_raw):
    freq = array(freq_raw)
    vals = interp.splev(array(freq)/1e6, narrow_acc_profile)
    vals[freq > 50e6] = interp.splev(50, narrow_acc_profile)
    vals[freq < 20e6] = interp.splev(20, narrow_acc_profile) * r(freq[freq < 20e6], 18e6, 2e6, 85e6, 10e6)
    return vals * r(freq, 0e6, 0e6, 70e6, 10e6) # arbitrary cut-off imposed, above experimentally tested frequencies
def transducer_efficiency_wide(freq_raw):
    freq = array(freq_raw)
    vals = interp.splev(array(freq)/1e6, wide_acc_profile)
    vals[freq > 50e6] = interp.splev(50, wide_acc_profile)
    vals[freq < 20e6] = interp.splev(20, wide_acc_profile)
    return vals * r(freq, 14e6, 7e6, 60e6, 10e6)

def make_aod_wide(orientation, ac_dir):
    """Create an Aod instance with a 3.3mm transducer. """
    return Aod(orientation, ac_dir, 16e-3, 3.25e-3, 8e-3, transducer_efficiency_wide)
def make_aod_narrow(orientation, ac_dir):
    """Create an Aod instance with a 1.2mm transducer. """
    return Aod(orientation, ac_dir, 16e-3, 1.15e-3, 8e-3, transducer_efficiency_narrow)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    f = linspace(10, 60, 300) * 1e6
    plt.figure()
    plt.plot(f/1e6, transducer_efficiency_narrow(f))
    plt.plot(f/1e6, transducer_efficiency_wide(f))
    plt.plot(f/1e6, f*0)
    plt.xlabel('freq / MHz')
    plt.ylabel('transducer efficiency')
