"""Helper functions to create rays, AODs, and AOLs."""

from aol_model.aol_full import AolFull
from aol_model.aod import Aod
from aol_model.ray import Ray
from numpy import array, linspace, exp, power, meshgrid, cos, sin
from scipy.constants import pi
from aol_model.vector_utils import normalise_list

# experimentally inferred data
import scipy.interpolate as interp
freq_points = [20,22,23,24,25,27,28,30,33,35,37,38,39,40,41,42,43,45,47,50]
profile_points = [0.52234631903519813, 0.55012067968962641, 0.6192829085994106, 0.68639957236869553, 0.63196874587441343, 0.47272087689723197, 0.42567446015841842, 0.48441767099699601, 0.65773072824321832, 0.77243458378400831, 0.81959932917417899, 0.89264810013635221, 0.92508418820625138, 0.90591507662766879, 0.84641841772725646, 0.82375973725822105, 0.80161022883030675, 0.74821649842319204, 0.68957047831550022, 0.65160805318284287]
acc_profile = interp.splrep(freq_points, profile_points)

def set_up_aol( op_wavelength, \
                order=-1, \
                base_freq=40e6, \
                focus_position=[0,0,1e12], \
                focus_velocity=[0,0,0], \
                pair_deflection_ratio=1, \
                ac_power=[1.5,1.5,2,2]):
    """Create an AolFull instance complete with Aods. """
    orient_39_920 = normalise_list(array([ \
        [-0.036, 0., 1], \
        [-0.054, -0.036,  1], \
        [-0.020, -0.054,  1], \
        [0.0, -0.026, 1] ])) # 0.022

    aod_spacing = array([5e-2] * 3)
    aods = [0]*4
    orientations = orient_39_920
    aods[0] = make_aod_wide(orientations[0], [1,0,0])
    aods[1] = make_aod_wide(orientations[1], [0,1,0])
    aods[2] = make_aod_narrow(orientations[2], [-1,0,0])
    aods[3] = make_aod_narrow(orientations[3], [0,-1,0])

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

import expt_data as data
import scipy.interpolate as interp
narrow_profile_points = [0.54712759754904783, 0.575386586772488, 0.64817850300475155, 0.71968269967894605, 0.66290982971563983, 0.49579761657367089, 0.44650474031596715, 0.50940265439071475, 0.69301186826603234, 0.81193150290459226, 0.85782888217413467, 0.93219904972501266, 0.96421680546132216, 0.94353454724247032, 0.88135434188178652, 0.8582870021491632, 0.83612955880696582, 0.78199419058440867, 0.72104707424642145, 0.68040177140906999]
narrow_acc_profile = interp.splrep(data.freq_narrow_new, narrow_profile_points)
wide_profile_points = [0.094350627384094748, 0.25152444812354441, 0.20224064418336243, 0.2149465067006722, 0.2210201524375898, 0.28646854606515304, 0.37895636626149154, 0.5169059716247596, 0.77216739537957224, 0.74133769980752928, 0.74696461830104255, 0.76521892972101946, 0.771015715779509, 0.75897803272828135, 0.74232442605432569, 0.77943013556525065, 0.73577720729480833, 0.67265427728068794, 0.61626958165703483, 0.42166897973248169]
wide_acc_profile = interp.splrep(data.freq_wide_new, wide_profile_points)

def transducer_efficiency_narrow(freq_raw):
    freq = array(freq_raw)
    vals = interp.splev(array(freq)/1e6, narrow_acc_profile)
    vals[freq > 50e6] = interp.splev(50, narrow_acc_profile)
    vals[freq < 20e6] = interp.splev(20, narrow_acc_profile)
    return vals * r(freq, 16e6, 5e6, 85e6, 10e6)
def transducer_efficiency_wide(freq_raw):
    freq = array(freq_raw)
    vals = interp.splev(array(freq)/1e6, wide_acc_profile)
    vals[freq > 50e6] = interp.splev(50, wide_acc_profile)
    vals[freq < 20e6] = interp.splev(20, wide_acc_profile)
    return vals * r(freq, 14e6, 7e6, 60e6, 10e6)

def make_aod_wide(orientation, ac_dir):
    """Create an Aod instance with a 3.3mm transducer. """
    return Aod(orientation, ac_dir, 16e-3, 3.15e-3, 8e-3, transducer_efficiency_wide)
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
