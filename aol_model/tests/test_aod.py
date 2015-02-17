from aol_model.aod import Aod
from aol_model.ray import Ray
from numpy import sqrt, allclose, cross, dot
from aol_model.vector_utils import normalise

aod = Aod([0,0,1], [1,0,0], 1, 1, 1)

def test_on_axis_ray_displacement():
    rays = [Ray([0,0,0],[0,0,1],800e-9,1)]*5
    aod.move_ray_through_aod(rays)
    still_on_axis = allclose(cross([r.position for r in rays], [0,0,1]), [0,0,0])
    direction_unchanged = allclose(r.wavevector_unit, [0,0,1])
    assert still_on_axis and direction_unchanged

def test_off_axis_ray_displacement():
    wavevec = [17./145,0,144./145]
    rays = [Ray([0,0,0],wavevec,800e-9,1)]*5
    aod.move_ray_through_aod(rays)
    off_wavevector = not allclose(cross([r.position for r in rays], wavevec), [0,0,0])
    direction_unchanged = allclose([r.wavevector_unit for r in rays], wavevec)
    assert off_wavevector and direction_unchanged

def test_refractive_indices_match():
    wavelen = 800e-9    
    wavevec = [3./5,0,4./5]
    rays = [Ray([0,0,0],wavevec,wavelen,1)]*5
    n1 = aod.calc_refractive_indices_vectors([r.wavevector_unit for r in rays], wavelen) 
    n2 = aod.calc_refractive_indices_rays(rays)
    assert allclose(n1,n2)

def test_refracting_in_towards_normal():
    wavevec = [3./5,0,4./5]
    rays = [Ray([0,0,0],wavevec,800e-9,1)]*5
    aod.refract_in(rays)
    cosine_outside = dot(wavevec, aod.normal) 
    cosine_inside =  dot([r.wavevector_unit for r in rays], aod.normal)
    towards_normal = abs(cosine_outside) < abs(cosine_inside)
    not_reflected = cosine_outside * cosine_inside >= 0
    assert towards_normal.all() and not_reflected.all()

def test_walkoff_towards_axis():
    wavevec = normalise([0.01,0,1])
    rays = [Ray([0,0,0],wavevec,800e-9,1)]*2
    directions = aod.get_ray_direction_ord(rays)
    cosine_wavevec = dot(wavevec, aod.optic_axis) 
    cosine_dir =  dot(directions[0], aod.optic_axis)
    walkoff_to_axis = cosine_wavevec < cosine_dir
    assert walkoff_to_axis

def test_refracting_in_at_normal():
    wavevec = [0,0,1]
    rays = [Ray([0,0,0],wavevec,800e-9,1)]*5
    aod.refract_in(rays)
    assert allclose(wavevec, [r.wavevector_unit for r in rays])

def test_refracting_out_away_from_normal():
    wavevec = [17./145,0,144./145]
    rays = [Ray([0,0,0],wavevec,800e-9,1)]*5
    aod.refract_out(rays)
    cosine_outside =  dot([r.wavevector_unit for r in rays], aod.normal)
    cosine_inside = dot(wavevec, aod.normal)
    towards_normal = abs(cosine_outside) < abs(cosine_inside)
    not_reflected = cosine_outside * cosine_inside >= 0
    assert towards_normal.all() and not_reflected.all()

def test_refracting_out_at_normal():
    wavevec = [0,0,1]
    rays = [Ray([0,0,0],wavevec,800e-9,1)]*5
    aod.refract_out(rays)
    assert allclose(wavevec, [r.wavevector_unit for r in rays])

def test_refraction_in_out_no_change():
    wavevec = [3./5,0,4./5]
    rays = [Ray([0,0,0],wavevec,800e-9,1)]*5
    aod.refract_in(rays)
    aod.refract_out(rays)
    assert allclose([r.wavevector_unit for r in rays], [3./5,0,4./5], rtol=5e-3) # should be close but not the same since ext in, ord out

def test_acoustic_direction_trivial():
    direc = aod.acoustic_direction
    assert allclose(direc, [1,0,0])
    
def test_acoustic_sound_direction():
    aod_new = Aod([1,0,1]/sqrt(2), [1,0,0], 1, 1, 1)
    direc = aod_new.acoustic_direction
    assert allclose(direc, [1,0,-1]/sqrt(2))

if __name__ == "__main__":
    test_walkoff_towards_axis()
    