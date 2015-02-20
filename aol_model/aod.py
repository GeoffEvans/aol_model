"""The aod module contains the Aod class, representing an AOD."""

from teo2 import calc_refractive_indices
from xu_stroud_model import diffract_acousto_optically
from vector_utils import perpendicular_component_list, normalise_list, normalise, \
    angle_between_unit_vectors
from error_utils import check_is_unit_vector
from numpy import array, sqrt, arcsin, sin, cos, cross, dot, dtype, outer, power, allclose
from numpy.linalg import norm
from scipy.optimize import fsolve
import teo2

class Aod(object):
    """The Aod class represents an AOD. An Aod object can be used alone to
    simulate a single AOD or in a group of four in an AolFull to simulate an AOL."""
    def __init__(self, normal, relative_ac_dir, transducer_height, transducer_width, crystal_thickness, transducer_efficiency_func=lambda x: 1):
        self.normal = array(normal, dtype=dtype(float))
        self.relative_acoustic_direction = array(relative_ac_dir, dtype=dtype(float))
        self.transducer_height = transducer_height
        self.crystal_thickness = crystal_thickness
        self.transducer_width = transducer_width
        self.transducer_efficiency_func = transducer_efficiency_func

        check_is_unit_vector(normal)
        check_is_unit_vector(relative_ac_dir)

    @property
    def optic_axis(self):
        """Included for future extensions to the model that allow rotated AOD
        designs. For now, the optic axis is taken to be aligned with the normal."""
        return self.normal
    @property
    def acoustic_direction(self):
        """The AOD aperture is, as a first approximation, normal to [0,0,1] (the z-axis)
        and the relative_acoustic_direction is defined based on this. The actual
        acoustic_direction needs to be rotated according for the exact normal."""

        # three basis vectors
        z = array([0.,0.,1.])
        invariant = normalise( cross(self.relative_acoustic_direction, z) )
        t = cross(z,invariant)

        # how z transforms
        cosine = dot(z, self.normal)
        sine = dot(t, self.normal)

        # components: s = s1 inv + s2 z + s3 t
        s1 = dot(invariant,self.relative_acoustic_direction)
        s2 = dot(z,self.relative_acoustic_direction)
        s3 = dot(t,self.relative_acoustic_direction)

        sound_vector = s1 * invariant + (cosine * s2 - sine * s3) * z + (cosine * s3 + sine * s2) * t

        return sound_vector

    def propagate_ray(self, rays, local_acoustics, order):
        """Take a bunch of optic rays through the AOD by first refracting into
        the AOD, then diffracting acousto-optically, propagating through the
        AOD and finally refracting out. """
        # can only take local_acoustics because there is no centre or location on the AOD object
        tol = 0.5 * (10**-teo2.accuracy)
        assert allclose([r.wavelength_vac for r in rays], rays[0].wavelength_vac, rtol=0, atol=tol) # can only handle small range of wavelengths at a time
        self.refract_in(rays)
        diffract_acousto_optically(self, rays, local_acoustics, order)
        self.move_ray_through_aod(rays)
        self.refract_out(rays)

    def move_ray_through_aod(self, rays):
        directions = self.get_ray_direction_ord(rays)
        distances = self.crystal_thickness / dot(directions, self.normal)
        for m in range(len(rays)):
            rays[m].position += distances[m] * directions[m]

    def get_ray_direction_ord(self, rays):
        """Ignore relatively minor walkoff due to shape of indicatrix."""
        return array([r.wavevector_unit for r in rays])

    def calc_refractive_indices_vectors(self, unit_vectors, wavelength):
        angles_to_axis = angle_between_unit_vectors(unit_vectors, self.optic_axis)
        return calc_refractive_indices(angles_to_axis, wavelength)

    def calc_refractive_indices_rays(self, rays):
        wavevecs = [r.wavevector_unit for r in rays]
        return self.calc_refractive_indices_vectors(wavevecs, rays[0].wavelength_vac)

    def refract_in(self, rays):
        """Refract an optic ray into the AOD."""
        # get vectors perpendicular and parallel to normal
        wavelength = rays[0].wavelength_vac
        wavevecs = [r.wavevector_unit for r in rays]
        perpendicular_comps = perpendicular_component_list(wavevecs, self.normal)

        unit_perpendiculars = normalise_list(perpendicular_comps)

        sin_angles_in = sqrt( 1 - power(dot(wavevecs, self.optic_axis), 2.) )
        angle_guesses = arcsin(sin_angles_in / 2.26)

        def zero_func(angles_out):
            wavevector_unit = outer(cos(angles_out), self.normal) + (sin(angles_out) * unit_perpendiculars.T).T
            n_ext = self.calc_refractive_indices_vectors(wavevector_unit, wavelength)[0]
            return (n_ext * sin(angles_out)) - sin_angles_in

        angles = fsolve(zero_func, angle_guesses, band=(0,0))
        wavevector_unit = outer(cos(angles), self.normal) + (sin(angles) * unit_perpendiculars.transpose()).transpose()

        for r,w in zip(rays, wavevector_unit):
            r.wavevector_unit = w

    def refract_out(self, rays):
        """Refract an optic ray out of the AOD."""
        wavevecs = array([r.wavevector_unit for r in rays])
        n_ords = self.calc_refractive_indices_rays(rays)[1]
        perpendicular_comps = perpendicular_component_list((n_ords * wavevecs.T).T, self.normal)
        parallel_components = outer(sqrt( 1 - power(norm(perpendicular_comps, axis=1), 2.) ), self.normal)
        for m in range(len(rays)): # if this is throwing exceptions, probably total internal reflection
            rays[m].wavevector_unit = parallel_components[m] + perpendicular_comps[m]
