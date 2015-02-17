from aol_model.aod import Aod
from aol_model.ray import Ray
from aol_model.acoustics import Acoustics
from aol_model.xu_stroud_model import diffract_acousto_optically,diffract_by_wavevector_triangle,get_efficiency
from aol_model.vector_utils import normalise
import pytest
from numpy import allclose, all, array, pi
from random import random
from scipy import less_equal, greater_equal

acoustics = Acoustics(40e6)
aod = Aod([0,0,1], [1,0,0], 1e-3, 1e-3, 1e-3)
order = 1
wavelen = 800e-9

def test_efficiency_range():
    
    def eff_fun():
        v = normalise([random(),random(),10])
        rays_in = [Ray([0,0,0],v,wavelen,1)]
        rays_out = [Ray([0,0,0],v,wavelen,1)]
        wavevecs_in_mag = [r.wavevector_vac_mag for r in rays_in]
        wavevecs_in_unit = [r.wavevector_unit for r in rays_in]   
        wavevecs_out_mag = [r.wavevector_vac_mag for r in rays_out]
        wavevecs_out_unit = [r.wavevector_unit for r in rays_out]   
        return get_efficiency(aod, random(), wavevecs_in_mag, wavevecs_in_unit, wavevecs_out_mag, wavevecs_out_unit, [acoustics], (0,1))
        
    effs = [ eff_fun() for _ in range(100) ]
    assert all(less_equal(effs,1)) and all(greater_equal(effs,0))  

def test_order_sym():
    r1 = Ray([0,0,0],[-17./145,0,144./145],wavelen)
    r2 = Ray([0,0,0],[ 17./145,0,144./145],wavelen)
    
    diffract_acousto_optically(aod, [r1], [acoustics], -1)
    diffract_acousto_optically(aod, [r2], [acoustics], 1)
    
    opposite_xcomps = allclose(r1.wavevector_unit[0], -r2.wavevector_unit[0])
    assert allclose(r1.energy, r2.energy) and opposite_xcomps  

def test_wavevector_triangle():
    wavevec_unit = array([0,0,1])
    wavevec_mag = 2 * pi / wavelen  
    (wavevector_mismatch_mag, wavevectors_out_unit, wavevectors_vac_mag_out) = diffract_by_wavevector_triangle(aod, array([wavevec_unit]), [wavevec_mag], [acoustics], order, (0,1))
    k_i = wavevec_unit * wavevec_mag * aod.calc_refractive_indices_vectors([wavevec_unit], wavelen)[0][0]
    k_d = wavevectors_out_unit * wavevectors_vac_mag_out * aod.calc_refractive_indices_vectors(wavevectors_out_unit, wavelen)[1] # get ord branch, the first of
    K = acoustics.wavevector(aod) * order
    zero_sum = k_i + order * K + aod.normal * wavevector_mismatch_mag - k_d[0]
    assert allclose(zero_sum, 0, atol=0.2, rtol=0)

def test_setting_invalid_mode():
    with pytest.raises(ValueError):
        ray = Ray([0,0,0,], [0,0,1], wavelen)
        diffract_acousto_optically(aod, [ray], [acoustics], 2)
        
if __name__ == '__main__':
    test_wavevector_triangle()