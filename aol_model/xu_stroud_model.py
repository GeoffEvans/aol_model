"""The heart of the model is the acousto-optic interaction solution described in Xu and Stroud.
The xu_stroud_model module contains the functions to diffract an optic ray accordiong to the
Xu and Stroud theory."""

from numpy import dot, sin, sqrt, array, power, outer, abs, all, isnan
from scipy.constants import c, pi
from numpy.linalg import norm
from vector_utils import normalise_list

def diffract_acousto_optically(aod, rays, local_acoustics, order, ext_to_ord=True):
    """The top level function handles the diffraction and sets out details
    including possible polarisations (ordinary or exrtaordinary -> ordinary
    or extraordinary) and whether second order diffraction is included."""

    if not abs(order) == 1:
        raise ValueError("Order only supports +1, -1")

    ref_inds = (1,0) # ord->ext
    if ext_to_ord:
        ref_inds = (0,1) # ext->ord

    wavevecs_in_mag  = [r.wavevector_vac_mag for r in rays] # store these
    wavevecs_in_unit  = array([r.wavevector_unit for r in rays])

    (efficiencies, wavevecs_out_unit, wavevecs_out_mag) = \
        get_diffracted_wavevectors_and_efficiency(aod, wavevecs_in_unit, wavevecs_in_mag, local_acoustics, order, ref_inds)

    # rescattering
    rev_ref_inds = ref_inds[::-1]
    (efficiencies_r,_,_) = get_diffracted_wavevectors_and_efficiency(aod, wavevecs_out_unit, wavevecs_out_mag, local_acoustics, order, rev_ref_inds)
    rescattering_terms = 0.5 * efficiencies_r # 0.5 inferred from single AOD experiment, may depend on AOD design and optical wavelength
    efficiencies *= 1 - rescattering_terms

    for r, m, u, e, resc in zip(rays, wavevecs_out_mag, wavevecs_out_unit, efficiencies, rescattering_terms):
        r.wavevector_vac_mag = m
        r.wavevector_unit = u
        r.energy *= e
        r.resc = resc * e

def get_diffracted_wavevectors_and_efficiency(aod, wavevecs_in_unit, wavevecs_in_mag, local_acoustics, order, ref_inds):
    """The basic Xu and Stroud theory is implemented in this function."""
    (wavevector_mismatches_mag, wavevecs_out_unit, wavevecs_out_mag) = diffract_by_wavevector_triangle(aod, wavevecs_in_unit, wavevecs_in_mag, local_acoustics, order, ref_inds)
    efficiencies = get_efficiency(aod, wavevector_mismatches_mag, wavevecs_in_mag, wavevecs_in_unit, wavevecs_out_mag, wavevecs_out_unit, local_acoustics, ref_inds)
    return (efficiencies, wavevecs_out_unit, wavevecs_out_mag)

def diffract_by_wavevector_triangle(aod, wavevec_unit_in, wavevec_vac_mag_in, local_acoustics, order, ref_inds):
    """Wavevector matching between the incident optic, acoustic and diffracted
    optic is dealt with here. The diffracted wavevector including its direction
    is calculated here. """
    wavevectors_vac_mag_out = wavevec_vac_mag_in + (2 * pi / c) * array([a.frequency for a in local_acoustics]) # from w_out = w_in + w_ac
    resultants = get_resultant_wavevectors(aod, wavevec_unit_in, wavevec_vac_mag_in, local_acoustics, order, ref_inds)

    f = lambda k: ref_ind_ext_ord(aod, normalise_list(k), wavevec_vac_mag_in)[ref_inds[1]] # pass this function to be used recursively
    return triangle_solve(resultants, wavevectors_vac_mag_out, aod.normal, f)

def get_resultant_wavevectors(aod, wavevec_unit_in, wavevec_vac_mag_in, local_acoustics, order, ref_inds):
    """Calculate the sum of the incident optic and acoustic wavevectors."""
    n_in = ref_ind_ext_ord(aod, wavevec_unit_in, wavevec_vac_mag_in)[ref_inds[0]]
    wavevectors_in = (n_in * wavevec_vac_mag_in * wavevec_unit_in.T).T
    wavevectors_ac = outer(array([a.wavevector_mag for a in local_acoustics]), aod.acoustic_direction)
    return wavevectors_in + order * wavevectors_ac

def get_efficiency(aod, wavevector_mismatches_mag, wavevecs_in_mag, wavevecs_in_unit, wavevecs_out_mag, wavevecs_out_unit, acoustics, ref_inds):
    """Based on the calculated diffracted wavevector, the diffraction efficiency can be calculated."""
    amp = [a.amplitude(aod) for a in acoustics] * sqrt(aod.transducer_efficiency_func([a.frequency for a in acoustics])) # square root because transducer eff is in terms of power

    n_in = ref_ind_ext_ord(aod, wavevecs_in_unit, wavevecs_in_mag)[ref_inds[0]]
    n_out = ref_ind_ext_ord(aod, wavevecs_out_unit, wavevecs_out_mag)[ref_inds[1]]
    p = -0.12  # for P66' (see appendix p583)

    delta_n0 = -0.5 * power(n_in, 2.) * n_out * p * amp # Xu & St (2.128)
    delta_n1 = -0.5 * power(n_out, 2.) * n_in * p * amp
    v0 = - array(wavevecs_out_mag) * delta_n0 * aod.transducer_width / dot(wavevecs_out_unit, aod.normal)
    v1 = - array(wavevecs_in_mag)  * delta_n1 * aod.transducer_width / dot(wavevecs_in_unit , aod.normal)

    zeta = -0.5 * wavevector_mismatches_mag * aod.transducer_width
    sigma = sqrt(power(zeta, 2.) + v0*v1/4) # Xu&St (2.132)

    return v0*v1/4 * power((sin(sigma) / sigma), 2.) # Xu&St (2.134)

def ref_ind_ext_ord(aod, unit_vector, wavevector_vac):
    return aod.calc_refractive_indices_vectors(unit_vector, 2*pi/wavevector_vac[0])

def triangle_solve(sum_vector, base_length, normal, multiplier_func):

    def tail_rec_solve(sv):
        multiplier = multiplier_func(sv) # ord ref ind
        ratio = base_length * multiplier / norm(sv, axis=1) # desired_wavelength_in_crystal / current_wavelength
        if all(abs(ratio - 1) < 1e-6):
            return sv
        new_sum_vector = sv - outer(dot(sv, normal) * (1 - ratio), normal) # improve approx
        return tail_rec_solve(new_sum_vector)

    wavevec_out = precondition(sum_vector, base_length, normal, multiplier_func)

    wavevec_out = tail_rec_solve(wavevec_out)
    wavevectors_out_unit = normalise_list(wavevec_out)

    wavevector_mismatches_mag = dot(wavevec_out - sum_vector, normal)

    return (wavevector_mismatches_mag, wavevectors_out_unit, base_length)

def precondition(sum_vectors, base_length, normal, multiplier_func):
    # use of xyz could be slightly misleading: z taken to be aligned with norm
    r_xy = sum_vectors - outer(dot(normal, sum_vectors.transpose()), normal)
    multiplier = multiplier_func(sum_vectors)
    r_z = sqrt( (base_length * multiplier)**2 - norm(r_xy, axis=1)**2 )
    assert not any(isnan(r_z)) # has gone off the indicatrix

    wavevec_out = r_xy + outer(r_z, normal)
    return wavevec_out