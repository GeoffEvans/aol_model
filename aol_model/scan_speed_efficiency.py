from numpy import linspace, pi, sin, cos, array, arctan2, meshgrid
from numpy.linalg import norm
from aol_model.set_up_utils import get_ray_bundle, set_up_aol
from aol_model.plot_utils import multi_line_plot, generic_plot_surface_vals

op_wavelength = 920e-9
scan_ang = 70e-3
scan_range_rad = linspace(-scan_ang/2, scan_ang/2, 199)
scan_deg = scan_range_rad * 180/pi

def plot_lines(focal_length, dwell_many_ns):
    funcs = []
    for dwell in dwell_many_ns * 1e-9:
        vel = [-focal_length*scan_ang/dwell/512,0,0]
        funcs.append(create_efficiency_function_closure([0,0,focal_length], vel))

    labels = ["scan angle / deg", "efficiency"]
    lgnd = dwell_many_ns.astype(int)
    multi_line_plot(scan_deg, funcs, labels, lgnd, (min(scan_deg),max(scan_deg),0,0.3))

def plot_fov_surf(focal_length, vel):
    from numpy import vectorize
    orthogonal_deg = scan_range_rad * 180/pi
    orthogonal_distance = focal_length * scan_deg * pi/180
    scan_angle = arctan2(vel[1], vel[0])

    effs = []
    for dist in orthogonal_distance:
        focus_position = dist * array([-sin(scan_angle), cos(scan_angle), 0]) + [0,0,focal_length]
        func = create_efficiency_function_closure(focus_position, vel)
        vec_func = vectorize(func)
        effs.append(vec_func(scan_deg))

    labels = ["xangle / deg", "yangle / deg", "efficiency"]
    (x, y) = get_xy(scan_angle, scan_deg, orthogonal_deg)
    generic_plot_surface_vals(x, y, array(effs), labels)

def get_xy(scan_angle, scan_deg, orthogonal_deg):
    (scan_mesh, orthogonal_mesh) = meshgrid(scan_deg, orthogonal_deg)
    x = scan_mesh * cos(scan_angle) - orthogonal_mesh * sin(scan_angle)
    y = scan_mesh * sin(scan_angle) + orthogonal_mesh * cos(scan_angle)
    return (x, y)

def create_efficiency_function_closure(focus_position_tzero, focus_velocity):
    focal_length = focus_position_tzero[2]
    aol = set_up_aol(focus_position=focus_position_tzero, focus_velocity=focus_velocity, op_wavelength=op_wavelength)

    def func(scan_deg):
        scan_distance = focal_length * scan_deg * pi / 180
        time = scan_distance / norm(focus_velocity)
        return calculate_efficiency(aol, time)
    return func

def calculate_efficiency(aol, time):
    rays = get_ray_bundle(op_wavelength)

    (_,energies) = aol.propagate_to_distance_past_aol(rays, time, 0)
    energy = sum(energies[:,-1])
    ray_count = len(rays)

    return energy / ray_count

if __name__ == '__main__':
    f = 1
    dwell = array([4000,2000,1000,500,200,100,50])
    # -abs(array([[-1e2,0,0],[-2e2,0,0],[-3e2,0,0],[-5e2,0,0],[-8e2,0,0],[-2e3,0,0]]))
    plot_lines(f, dwell)
    #plot_fov_surf(1e1, 600 * array([1,0,0]))