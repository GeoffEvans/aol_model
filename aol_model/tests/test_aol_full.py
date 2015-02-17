from aol_model.aol_full import AolFull
from aol_model.aod import Aod
from aol_model.ray import Ray
from aol_model.vector_utils import normalise
from aol_model.aol_simple import AolSimple
from aol_model.acoustics import teo2_ac_vel
from numpy import allclose, array, arange, outer, linspace, meshgrid, dot,\
    concatenate, mean, std
from random import random as r

order = 1
op_wavelength = 800e-9
base_freq = 40e6
pair_deflection_ratio = 0.8
crystal_thickness = 8e-3

focal_length = 1
focus_position = array([-.01,-.01,focal_length])
focus_velocity = [1,1,0]

aod_spacing = array([5e-2] * 3)

aods = [0]*4
aods[0] = Aod(normalise([0,1,40]), [ 1, 0,0], 25e-3, 3.2e-3, crystal_thickness)
aods[1] = Aod(normalise([1,-1,40]), [ 0, 1,0], 25e-3, 3.2e-3, crystal_thickness)
aods[2] = Aod([0,0,1], [-1, 0,0], 25e-3, 1.6e-3, crystal_thickness)
aods[3] = Aod([0,0,1], [ 0,-1,0], 25e-3, 1.6e-3, crystal_thickness)

aol = AolFull.create_aol(aods, aod_spacing, order, op_wavelength, base_freq, pair_deflection_ratio, focus_position, focus_velocity)
aol_simple = AolSimple.create_aol(order, op_wavelength, teo2_ac_vel, aod_spacing, base_freq, pair_deflection_ratio, focus_position, focus_velocity, [crystal_thickness]*4)

def test_aol_drives_same():
    const_full = [a.const for a in aol.acoustic_drives]
    const_simple = [a.const for a in aol_simple.acoustic_drives]
    linear_full = [a.linear for a in aol.acoustic_drives]
    linear_simple = [a.linear for a in aol_simple.acoustic_drives]
    quad_full = [a.quad for a in aol.acoustic_drives]
    quad_simple = [a.quad for a in aol_simple.acoustic_drives]
    assert allclose(const_full, const_simple) and allclose(linear_full, linear_simple) and allclose(quad_full, quad_simple)

def test_plot():
    import matplotlib.pyplot as plt
    x,y = meshgrid(linspace(-1,1,5)*1e-2, linspace(-1,1,5)*1e-2)
    list_of_positions = zip(x.ravel(), y.ravel()) 
    rays = [Ray([xy[0], xy[1], 0], [0,0,1], op_wavelength) for xy in list_of_positions]
    plt.ion()
    aol.plot_ray_through_aol(rays, 0, focus_position[2])
    plt.close()

def test_angles_on_aods():
    x,y = meshgrid(linspace(-1,1,5)*1e-2, linspace(-1,1,5)*1e-2)
    list_of_positions = zip(x.ravel(), y.ravel()) 
    rays = [Ray([xy[0], xy[1], 0], [0,0,1], op_wavelength) for xy in list_of_positions]
    paths,_ = aol.propagate_to_distance_past_aol(rays, 3e-6)
    for m in range(8):
        dot_prods = dot(paths[:,m,:], aol.aods[m/2].normal)
        assert allclose(dot_prods, dot_prods[0])
    assert allclose([p[2] for p in paths[:,8,:]], aod_spacing.sum())

def test_ray_passes_through_focus():
    location = [0]*100
    for t in arange(100):
        ray = Ray([r()*5e-2,r()*5e-2,0], [0,0,1], op_wavelength)
        aol.propagate_to_distance_past_aol([ray], 0, focal_length)
        location[t] = ray.position
    
    focus_theory = focus_position + concatenate( (aol_simple.base_ray_positions[3], [aod_spacing.sum()]) ) 
    assert allclose(mean(location, axis=0), focus_theory, rtol=0, atol=1e-3) \
            and all(std(location, axis=0) < 5e-5)

def test_ray_scans_correctly():
    t_step = 1e-6
    num_rays = 100
    location = [0]*num_rays
    for t in arange(num_rays):
        ray = Ray([r()*5e-2,r()*5e-2,0], [0,0,1], op_wavelength)
        aol.propagate_to_distance_past_aol([ray], t*t_step, focal_length)
        location[t] = ray.position
    
    focus_theory = focus_position + concatenate( (aol_simple.base_ray_positions[3], [aod_spacing.sum()]) ) + outer(arange(num_rays)*t_step, focus_velocity)
    assert allclose(location, focus_theory, rtol=0, atol=2e-3)

def test_efficiency_low_at_angle():
    ray = Ray([0,0,0], [3./5,0,4./5], op_wavelength)
    aol.propagate_to_distance_past_aol([ray], 0, focal_length)
    assert ray.energy < 1e-9
    
if __name__ == '__main__':
    #test_ray_passes_through_focus()
    test_angles_on_aods()