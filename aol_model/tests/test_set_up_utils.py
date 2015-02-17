import aol_model.set_up_utils as su

op_wavelength = 800e-9

# all ok as long as no exception 

def test_ray_bundle_and_aol():
    rays = su.get_ray_bundle(op_wavelength=op_wavelength)
    aol = su.set_up_aol(op_wavelength, 1, 40e6, [1e-3,2e-3,1], [1e3,5e2,0], 0.2, [2,2,2,2])
    aol.propagate_to_distance_past_aol(rays, 0, 10)
    
def test_transducer_freqs():
    freq = 40e6
    su.transducer_efficiency_narrow(freq)
    su.transducer_efficiency_narrow2(freq)
    su.transducer_efficiency_wide(freq)

def test_aod_making():
    orientation = [0,0,1]
    ac_dir = [1,0,0]
    su.make_aod_wide(orientation, ac_dir)
    su.make_aod_narrow(orientation, ac_dir)
    su.make_aod_narrow2(orientation, ac_dir)

if __name__ == '__main__':
    test_ray_bundle_and_aol()