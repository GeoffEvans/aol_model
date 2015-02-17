from aol_model.optimise_aol import calculate_efficiency
from aol_model.set_up_utils import set_up_aol

def test_efficiency():
    aol = set_up_aol(800e-9)
    eff = calculate_efficiency(aol, 4)
    
    assert eff < 1 and eff > 0

test_efficiency()