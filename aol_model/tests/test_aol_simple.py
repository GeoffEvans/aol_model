from aol_model.aol_simple import AolSimple
from aol_model.ray import Ray
import pytest
from numpy import allclose, array

wavelength = 800e-9
order = -1
spacing = [1,1,1]

def test_non_unit_direction():
    with pytest.raises(ValueError):
        AolSimple(1, [1,1,1], [0]*4, [0]*4, aod_directions=[[1,0,0.1],[0,1,0],[-1,0,0],[0,-1,0]])
        
def test_plot():
    import matplotlib.pyplot as plt
    aol = AolSimple.create_aol_from_drive(order, spacing, array([1]*4)*1e6, [1]*4, wavelength)
    wavevec = [0,3./5,4./5]
    ray = Ray([0,0,0], wavevec, wavelength)
    plt.ion()
    aol.plot_ray_through_aol(ray, 1, aol.aod_spacing.sum())
    plt.close()
    assert allclose(ray.position, [0,0,0], atol=0) and allclose(ray.wavevector_unit, wavevec, atol=0)

def test_no_chirp_at_tzero():
    aol = AolSimple.create_aol_from_drive(order, spacing, [0]*4, array([1e6]*4), wavelength)
    aol.set_base_ray_positions(wavelength)
    wavevec = [0,0,1]
    ray = Ray([0,0,0], wavevec, wavelength)
    aol.propagate_to_distance_past_aol(ray, 0, 10)
    assert allclose(ray.wavevector_unit, wavevec, atol=0)
    
def test_constant_freq_for_zero_chirp():
    aol = AolSimple.create_aol_from_drive(order, spacing, [10]*4, [0]*4, wavelength)
    wavevec = [0,3./5,4./5]
    ray1 = Ray([0,0,0], wavevec, wavelength)
    aol.propagate_to_distance_past_aol(ray1, 0)
    ray2 = Ray([0,0,0], wavevec, wavelength)
    aol.propagate_to_distance_past_aol(ray2, 1e-3)
    assert allclose(ray1.wavevector_unit, ray2.wavevector_unit, atol=0)    
    
def test_deflect_right_way():
    aol = AolSimple.create_aol_from_drive(order, spacing, [1,0,0,0], [0]*4, wavelength)
    wavevec = [0,3./5,4./5]
    ray = Ray([0,0,0], wavevec, wavelength)
    aol.propagate_to_distance_past_aol(ray, 0)
    assert ray.wavevector_unit[0] < 0    
    
def test_find_base_ray_positions():
    aol_const = AolSimple.create_aol_from_drive(order, spacing, [1e6]*4, [0]*4, wavelength)
    aol_chirp = AolSimple.create_aol_from_drive(order, spacing, [1e6]*4, [1e6]*4, wavelength)
    assert allclose(aol_const.base_ray_positions, aol_chirp.base_ray_positions, atol=0) and not aol_chirp.acoustic_drives[0].linear == 0 

if __name__ == '__main__':
    test_constant_freq_for_zero_chirp()
    test_plot()