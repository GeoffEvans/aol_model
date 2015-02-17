from error_utils import check_is_unit_vector
from numpy import pi, array, dot, dtype, concatenate
from numpy.linalg import norm

class Ray(object):
        
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
        check_is_unit_vector(v) # useful for error checking but slow!
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
        self.wavevector_vac_mag = norm(v)        
        self.wavevector_unit = array(v, dtype=dtype(float)) / self.wavevector_vac_mag
    
    def propagate_free_space(self, distance):
        self.position += self.wavevector_unit * distance
        
    def propagate_to_plane(self, point_on_plane, normal_to_plane):
        from_ray_to_point = point_on_plane - self.position
        distance = dot(from_ray_to_point, normal_to_plane) / dot(self.wavevector_unit, normal_to_plane)
        self.propagate_free_space(distance)
        
    def propagate_from_plane_to_plane(self, plane_z_separation, normal_to_first, normal_to_second):
        point_on_first_plane = self.position
        z_displacement_from_point_to_origin = dot(point_on_first_plane[0:2], normal_to_first[0:2]) / normal_to_first[2] 
        displacement_from_point_to_origin = concatenate( (-point_on_first_plane[0:2], [z_displacement_from_point_to_origin]) )
        
        # assumes all AODs are rotated about (x,y)=(0,0), in future would be faster and more realistic to use an AOD centre property
        point_on_second_plane = point_on_first_plane + displacement_from_point_to_origin + [0,0,plane_z_separation]
        self.propagate_to_plane(point_on_second_plane, normal_to_second)
        
    def propagate_free_space_z(self, distance):
        self.propagate_to_plane(self.position + [0,0,distance], [0,0,1])
