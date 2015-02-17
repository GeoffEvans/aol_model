from aol_model.ray_paraxial import RayParaxial
from numpy import allclose, array
import pytest

position = [0,0,2]
wavevector_unit = [0.1,0,1]
wavelength = 10
energy = 1

def test_setting_non_unit_z_component():
    with pytest.raises(ValueError):
        RayParaxial(position, [1,0,0.1], wavelength, energy)
    
def test_propagating_z():
    wavevec_local = array([1,2,1])
    r = RayParaxial(position, [1,2,1], wavelength, energy)
    r.propagate_free_space_z(5)
    assert allclose(r.position, position + 5*wavevec_local)

def test_setting_wavevector_property():
    r = RayParaxial(position, [0,1,1], wavelength, energy)
    mag = r.wavevector_vac_mag
    r.wavevector_vac = [1,0,0]
    mag_correct = allclose(r.wavevector_vac_mag, mag)
    dir_correct = allclose(r.wavevector_unit, [1./mag,0,1])
    assert mag_correct and dir_correct

