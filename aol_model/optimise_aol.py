"""The optimise_aol module is useful to calculate the correct orientation of each AOD in an AOL."""

from numpy import arange, linspace, pi, arctan2, array, cos, sin, sqrt, append
from numpy.linalg import norm
from scipy import optimize
from aod_visualisation import generic_plot_surface
from set_up_utils import set_up_aol, get_ray_bundle

op_wavelength = 909e-9
base_freq = 39e6
pdr = 0

class OptParams(object):
    """A class used to hold optimisation parameters."""
    def __init__(self, start, stop):
        self.xy_start_deg = start
        self.xy_end_deg = stop

    def create_line(self):
        start = array(self.xy_start_deg) * pi/180
        end = array(self.xy_end_deg) * pi/180
        min_val = 0
        fixed = start
        diff = end - start
        max_val = norm(diff)
        ang = arctan2(diff[1], diff[0])
        return (min_val, max_val, ang, fixed)

    def min_val(self):
        return self.create_line()[0]
    def max_val(self):
        return self.create_line()[1]
    def ang(self):
        return self.create_line()[2]
    def fixed(self):
        return self.create_line()[3]

    def get_normal(self, variable):
        (_, _, ang, fixed) = self.create_line()
        xy_normal = fixed + variable * array([cos(ang), sin(ang)])
        return append(xy_normal, sqrt(1 - norm(xy_normal)**2))

def plot_region(aod_num, aol):
    """For the given aol_num, plots AOD diffraction efficiency over a range of
    angles in both x and y. The plot should be used to supply the
    inputs required by optimise_nth_aod_by_hand: click on two points lying
    either side of the peak such that the line between them crosses the peak.
    The x,y angles will be printed to the console."""

    def func(scan_deg, y_deg):
        x = scan_deg * pi/180
        y = y_deg * pi/180
        new_normal = [x, y, sqrt(1 - x**2 - y**2)]
        aol.change_orientation(aod_num, new_normal)
        energies = calculate_efficiency(aol, aod_num)
        return energies

    labels = ["incidence angle / deg","transverse incidence angle / deg","efficiency"]
    x_ax = linspace(-0.05, 0.05, 20)*180/pi
    y_ax = linspace(-0.2, 0.2, 30)*180/pi
    if aod_num % 2 == 0:
        temp = x_ax
        x_ax = y_ax
        y_ax = temp

    generic_plot_surface(x_ax, y_ax, func, labels)

def optimise_nth_aod_by_hand(aod_num, aol):
    """Calculate the optimal AOD angle and corresponding efficiency by searching
    along a line for the given aol_num. The ends of the line are taken as two
    separate console inputs and are both specified as [x_angle, y_angle]. Both
    pairs of [x_angle, y_angle] should be read off a plot produced by plot_region.
    The return value must be copied into the set_up_utils to be used as the
    AOD orientations"""

    print 'enter start'
    start = input()
    print 'enter stop'
    stop = input()
    p = OptParams(start, stop)

    result = optimize.fminbound(min_fun, p.min_val(), p.max_val(), (p, aod_num, aol), full_output=True)
    new_optimal_normal = p.get_normal(result[0])
    aol.change_orientation(aod_num, new_optimal_normal)
    print [i for i in new_optimal_normal]

def min_fun(variable, params, aod_num, aol):
    new_normal = params.get_normal(variable)
    aol.change_orientation(aod_num, new_normal)
    return - calculate_efficiency(aol, aod_num)

def calculate_efficiency(aol, after_nth_aod, op_wavelength=op_wavelength):
    """Calculate the efficiency of the AOL up to and including
    the (after_nth_aod) AOD."""

    time_array = (arange(3)-1)*5e-5

    energy = 0
    ray_count = 0

    for t in time_array:
        rays = get_ray_bundle(op_wavelength)

        (_,energies) = aol.propagate_to_distance_past_aol(rays, t)
        energy += sum(energies[:,after_nth_aod-1])
        ray_count += len(rays)

    return energy / ray_count

def get_best_pdr_x(pdr, ang):
    aol = set_up_aol(op_wavelength, base_freq=base_freq, pair_deflection_ratio=pdr, focus_position=[ang*3.14159/180*1e9,0,1e9])
    return -calculate_efficiency(aol, 1) * calculate_efficiency(aol, 3) / calculate_efficiency(aol, 2)

def get_best_pdr_y(pdr, ang):
    aol = set_up_aol(op_wavelength, base_freq=base_freq, pair_deflection_ratio=pdr, focus_position=[ang*3.14159/180*1e9,0,1e9])
    return -calculate_efficiency(aol, 2) * calculate_efficiency(aol, 4) / calculate_efficiency(aol, 3)

if __name__ == '__main__':
    aol = set_up_aol(op_wavelength, base_freq=base_freq, pair_deflection_ratio=-0.4, focus_position=[1.1*3.14159/180*1e12,0,1e12])
    print calculate_efficiency(aol, 1)
    print calculate_efficiency(aol, 2)/calculate_efficiency(aol, 1)
    print calculate_efficiency(aol, 3)/calculate_efficiency(aol, 2)
    print calculate_efficiency(aol, 4) # check efficiency at AOD


    #import matplotlib.pyplot as plt
    #import scipy.optimize as opt
    #list_ang = arange(-0.6, 2, 0.1)
    #pdrs = []
    #for ang in list_ang:
    #    pdrs.append(opt.minimize_scalar(get_best_pdr_x, bounds=(-0.5, 2), method='bounded', args=(ang,)).x)
    #plt.plot(list_ang, pdrs)

    #plot_region(1, aol) # make plot, click on two points both lying on the same line through the peak
    #optimise_nth_aod_by_hand(1, aol) # optimise aod using previous plot
