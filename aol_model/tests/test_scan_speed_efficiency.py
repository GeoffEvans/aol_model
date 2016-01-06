from aol_model.set_up_utils import set_up_aol
from aol_model.scan_speed_efficiency import calculate_efficiency

op_wavelength = 800e-9
aol = set_up_aol(op_wavelength=op_wavelength, focus_position=[0,0,1])

def test_calculate_efficiency():
    e0 = calculate_efficiency(aol, 0)
    e1 = calculate_efficiency(aol, 1e-4)
    assert e0 > 1e-1 and e1 < 1e-1

if __name__ == '__main__':
    test_calculate_efficiency()