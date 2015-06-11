from plot_utils import multi_line_plot_vals
from numpy import linspace, shape, pi, array, meshgrid, arange, prod, transpose, power, max
from set_up_utils import get_ray_bundle, set_up_aol
import matplotlib.pyplot as plt
from scipy import ndimage
from matplotlib import rcParams as r
r.update({'font.size': 24})

op_wavelength = 920e-9
base_freq = 39e6

x_rad = linspace(-36, 36, 50) * 1e-3
x_deg = x_rad * 180/pi

def plot_fov_lines(focal_lengths, pdr):
    """Plot field of view across y_angle=0 for a list of focal lengths and a fixed pair deflection ratio. """
    focus_position_many = []
    for f in focal_lengths:
        x = f * x_rad
        focus_position_many.append( array([x, 0*x, f+0*x]) )
    effs = get_effs(transpose(focus_position_many, [0,2,1]), pdr)

    labels = ["xangle / deg", "efficiency"]
    multi_line_plot_vals(x_deg, array(effs), labels, array(focal_lengths).astype(int), (min(x_deg),max(x_deg),0,1))

def plot_fov_surf(focal_length, pdr):
    """Plot field of view over a surface of solid angle (x_angle and y_angle) for a single focal length and a fixed pair deflection ratio. """
    effs_norm = calc_fov_surf_data(focal_length, pdr)
    description = 'Model for PDR %s' % pdr
    generate_plot(effs_norm, description)
    return effs_norm

def calc_fov_surf_data(focal_length, pdr):
    (x_deg_m, y_deg_m) = meshgrid(x_deg, x_deg)
    x_array = x_rad * focal_length
    (x, y) = meshgrid(x_array, x_array)
    focus_position_many = transpose(array([x, y, focal_length+0*x]), [1,2,0])

    effs = get_effs(focus_position_many, pdr)
    effs_norm = effs / max(effs)
    return effs_norm

def plot_peak(focal_lengths):
    import matplotlib.pyplot as plt
    """Plot peak efficiency for a range of focal lenghts. Pair deflection ratio independent. """
    focus_position_many = array([[[0,0,f]] for f in focal_lengths])
    effs = get_effs(focus_position_many, 0.3) # the 0.3 here is an arbitrary pdr value
    z = 1/array(focal_lengths)
    labels = ["1/z / 1/m", "efficiency"]
    plt.plot(z, array(effs)/max(effs), label=labels, marker='x')
    plt.axis((min(z),max(z),0,1))

def generate_plot(normalised_img, description, colmap=plt.cm.bone, pdr_z=None):
    fig = plt.figure()
    angles = linspace(-36, 36, shape(normalised_img)[0]) * 1e-3 * 180/pi

    cm = plt.pcolormesh(angles, angles, normalised_img, cmap=colmap, vmin=0, vmax=1)
    cm.set_rasterized(True)
    plt.xticks([-2,0,2], ['','',''])
    plt.xlim([-2,2])
    plt.ylim([-2,2])
    plt.yticks([-2,0,2], ['','',''])
    plt.tick_params(direction='out')
    #plt.colorbar()

    #has_contour = 0
    #img_blur = ndimage.gaussian_filter(normalised_img, 1)
    #cset = plt.contour(angles, angles, img_blur, arange(0.3,1,0.2), linewidths=has_contour, cmap=plt.cm.coolwarm)
    #plt.clabel(cset, inline=True, fmt='%1.1f', fontsize=20)

    labels = ["x angle / deg", "y angle / deg", "efficiency"]
    ax = fig.gca()
    #ax.set_xlabel(labels[0])
    #ax.set_ylabel(labels[1])
    #ax.text(0.03, 0.9, description, transform=ax.transAxes, color='w', fontsize=27)
    ax.set_aspect('equal', adjustable='box')


    if False and pdr_z is not None and pdr_z[0] is not None and pdr_z[0] > -1:
        #ang = - 9e6 * (920e-9 / 613) * (1+pdr) * (180 / pi)
        L = 0.01
        r = pdr_z[0]
        z = pdr_z[1]
        ang = -18 * (L*r/z + r + 1) * 0.92/613 / (2.15 + r) * 180 / pi
        if ang < -2: # all ok
            return
        if ang > 2: # none ok
            ang = 1.7
        plt.plot([ang, ang], [ang, 2], 'y--', linewidth=4)
        plt.plot([ang, 2], [ang, ang], 'y--', linewidth=4)

def get_effs(focus_position_many, pdr):
    #get eff for a 2d array for focus positions (so 3d array input)
    shp = focus_position_many.shape[0:2]
    aols = [set_up_aol(op_wavelength, focus_position=f, base_freq=base_freq, pair_deflection_ratio=pdr) for f in focus_position_many.reshape(prod(shp), 3)]
    effs = [calculate_efficiency(a) for a in aols]
    return array(effs).reshape(shp)

def calculate_efficiency(aol):
    time_array = (arange(3)-1)*1e-7
    energy = 0
    ray_count = 0

    for time in time_array:
        rays = get_ray_bundle(op_wavelength)
        (_,energies) = aol.propagate_to_distance_past_aol(rays, time, 0)
        energy += sum(energies[:,-1])
        ray_count += len(rays)

    return power(energy / ray_count, 2)

if __name__ == '__main__':
    effs = plot_fov_surf(1e9, 0.3)
    #print max(effs)
    #plot_peak([-0.2, -0.25, -0.33, -0.4, -0.5, -1, -1e3, 1e3, 1, 0.5, 0.4, 0.33, 0.25, 0.2])
