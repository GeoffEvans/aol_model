from aol_model.acoustics import AcousticDrive
from numpy import allclose

def test_only_const():
    drive = AcousticDrive(10, 0)
    f = [0]*4
    f[1] = drive.get_local_acoustics(0, [[0,0,0]], [1,1], [1,0,0])[0].frequency
    f[2] = drive.get_local_acoustics(10, [[0,0,0]], [1,1], [1,0,0])[0].frequency
    f[3] = drive.get_local_acoustics(10, [[1,1,1]], [1,1], [1,0,0])[0].frequency
    f[0] = drive.get_local_acoustics(10, [[1,1,1]], [1,1], [0,0,1])[0].frequency
    
    assert allclose(f,f[0])

def test_no_offset():
    drive = AcousticDrive(10, 100)
    f = drive.get_local_acoustics(0, [[2,3,1]], [2,3], [1,0,0])[0].frequency
    assert f == 10
    
def test_x_shift():
    drive = AcousticDrive(1000, 10, velocity=1, ramp_time=100)
    f1 = drive.get_local_acoustics(0, [[0,0,0]], [0,0], [1,0,0])[0].frequency
    f2 = drive.get_local_acoustics(0, [[10.,0,0]], [0,0], [1,0,0])[0].frequency
    
    assert allclose(f2 - f1, -100)
    
def test_base_shift():
    drive = AcousticDrive(1000, 10, velocity=1, ramp_time=100)
    f1 = drive.get_local_acoustics(0, [[0,0,0]], [0,0], [1,0,0])[0].frequency
    f2 = drive.get_local_acoustics(0, [[0,0,0]], [-10.,0], [1,0,0])[0].frequency
    
    assert allclose(f2 - f1, -100)
    
def test_t_shift():
    drive = AcousticDrive(10, 10, ramp_time=100)
    f1 = drive.get_local_acoustics(0, [[0,0,0]], [0,0], [1,0,0])[0].frequency
    f2 = drive.get_local_acoustics(10., [[0,0,0]], [0,0], [1,0,0])[0].frequency
    
    assert allclose(f2 - f1, 100)

def test_ramp_loop():
    drive = AcousticDrive(10, 10, ramp_time=10)
    f1 = drive.get_local_acoustics(0, [[0,0,0]], [0,0], [1,0,0])[0].frequency
    f2 = drive.get_local_acoustics(10., [[0,0,0]], [0,0], [1,0,0])[0].frequency
    
    assert allclose(f2 - f1, 0)
    
if __name__ == '__main__':
    test_ramp_loop()