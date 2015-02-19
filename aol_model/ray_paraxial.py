from numpy import pi, array, dtype
from error_utils import check_is_val

class RayParaxial(object):
    """A simplified ray whose angle to [0,0,1] (z-axis) is assumed to be small. """
    def __init__(self, position, wavevector_unit, wavelength, energy=1):
        self.position = array(position, dtype=dtype(float))
        self.wavevector_unit = array(wavevector_unit, dtype=dtype(float))
        self.wavelength_vac = wavelength
        self.energy = energy

    @property
    def wavevector_unit(self):
        return self._wavevector_unit
    @wavevector_unit.setter
    def wavevector_unit(self, v):
        check_is_val(v[2], 1)
        self._wavevector_unit = array(v, dtype=dtype(float))

    @property
    def wavevector_vac_mag(self):
        return 2 * pi / self.wavelength_vac
    @wavevector_vac_mag.setter
    def wavevector_vac_mag(self, v):
        self.wavelength_vac = 2 * pi / v

    @property
    def wavevector_vac(self):
        return self.wavevector_vac_mag * self.wavevector_unit
    @wavevector_vac.setter
    def wavevector_vac(self, v):
        self.wavevector_unit[0:2] = array(v[0:2]) / self.wavevector_vac_mag

    def propagate_free_space_z(self, distance):
        """Move the ray in the direction of its wavevector such that it travels
        a given distance in z."""
        self.position += self.wavevector_unit * distance
