from aol_model.teo2 import calc_refractive_indices, get_relative_impermeability_eigenvals
from numpy import pi, arange, allclose, array, power

wavelen = 800e-9

def test_ord_less_than_ext():
    angles = arange(0,pi/2,pi/10)
    refractive_indices = calc_refractive_indices(angles, wavelen)

    ord_less_than_ext = refractive_indices[1] < refractive_indices[0]
    assert ord_less_than_ext.all()

def test_extreme_ref_vals():
    angles = array([0,pi/2])
    
    refractive_indices = calc_refractive_indices(angles, wavelen)
    
    n_e_vals = [2.226, 2.373]
    n_o_vals = [2.226, 2.226]
    
    n_e_eq = allclose(refractive_indices[0], n_e_vals, atol=1e-2)
    n_o_eq = allclose(refractive_indices[1], n_o_vals, atol=1e-2)
    
    return n_o_eq and n_e_eq
    
def test_symmetry():
   
    def all_elements_same(lst):
        return allclose(lst, lst[0], atol=1e-15)
   
    angles = array([0.3,-0.3])
    refractive_indices = calc_refractive_indices(angles, wavelen)
    
    ext_same = all_elements_same(refractive_indices[0])
    ord_same = all_elements_same(refractive_indices[1])
    
    assert ext_same and ord_same 
    
def test_single_angle():
    angles = 0.1
    calc_refractive_indices(angles, wavelen) # don't want this to throw
    
def test_relative_impermeability_eigenvals():
    wavs = [0.4047e-6, 0.6328e-6, 1e06]
    ref_inds = array([power(get_relative_impermeability_eigenvals(w), -0.5) for w in wavs])
    assert allclose(ref_inds[:,0], ref_inds[:,1])
    assert allclose(ref_inds[:,0], [2.4315, 2.2597, 2.208], atol=0.05) #ords
    assert allclose(ref_inds[:,2], [2.6157, 2.4119, 2.352], atol=0.05) #exts