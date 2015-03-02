"""Helper functions to create rays, AODs, and AOLs."""

from aol_model.aol_full import AolFull
from aol_model.aod import Aod
from aol_model.ray import Ray
from numpy import array, linspace, exp, power, meshgrid, cos, sin
from scipy.constants import pi
from aol_model.vector_utils import normalise_list

def set_up_aol( op_wavelength, \
                order=-1, \
                base_freq=40e6, \
                focus_position=[0,0,1e12], \
                focus_velocity=[0,0,0], \
                pair_deflection_ratio=1, \
                ac_power=[1,1,2,2]):
    """Create an AolFull instance complete with Aods. """
    orient_39_920 = normalise_list(array([ \
        [-0.036, 0., 1], \
        [-0.054, -0.036,  1], \
        [-0.022, -0.054,  1], \
        [0.0, -0.022, 1] ]))

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

def transducer_efficiency_narrow(freq):
    return r(array(freq), 13e6, 10e6, 90e6, 10e6)
def transducer_efficiency_wide(freq):
    return r(array(freq), 15e6, 10e6, 85e6, 10e6)

def make_aod_wide(orientation, ac_dir):
    """Create an Aod instance with a 3.3mm transducer. """
    return Aod(orientation, ac_dir, 16e-3, 3.3e-3, 8e-3, transducer_efficiency_wide)
def make_aod_narrow(orientation, ac_dir):
    """Create an Aod instance with a 1.2mm transducer. """
    return Aod(orientation, ac_dir, 16e-3, 1.2e-3, 8e-3, transducer_efficiency_narrow)