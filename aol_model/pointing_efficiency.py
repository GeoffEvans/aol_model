from plot_utils import multi_line_plot_vals
from numpy import linspace, shape, pi, array, meshgrid, arange, prod, transpose, power, max
from set_up_utils import get_ray_bundle, set_up_aol
import matplotlib.pyplot as plt
from matplotlib import rcParams as r
r.update({'font.size': 30})

op_wavelength = 920e-9
base_freq = 39e6

x_rad = linspace(-36, 36, 20) * 1e-3
x_deg = x_rad * 180/pi

def plot_fov_lines(focal_lengths, pdr):
    focus_position_many = []
    for f in focal_lengths:
        x = f * x_rad
        focus_position_many.append( array([x, 0*x, f+0*x]) )
    effs = get_effs(transpose(focus_position_many, [0,2,1]), pdr)
    
    labels = ["xangle / deg", "efficiency"]
    multi_line_plot_vals(x_deg, array(effs), labels, array(focal_lengths).astype(int), (min(x_deg),max(x_deg),0,1))
        
def plot_fov_surf(focal_length, pdr):    
    (x_deg_m, y_deg_m) = meshgrid(x_deg, x_deg) 

    x_array = x_rad * focal_length
    (x, y) = meshgrid(x_array, x_array)
    focus_position_many = transpose(array([x, y, focal_length+0*x]), [1,2,0])
    
    effs = get_effs(focus_position_many, pdr)
    effs_norm = effs / max(effs)
    
    description = 'Model for PDR %s' % pdr
    generate_plot(effs, effs_norm, description)
    return effs_norm
    
def generate_plot(orig_img, normalised_img, description, colmap=plt.cm.bone):
    fig = plt.figure()
    angles = linspace(-36, 36, shape(orig_img)[0]) * 1e-3 * 180/pi
        
    plt.pcolormesh(angles, angles, orig_img, cmap=colmap)
    
    cset = plt.contour(angles, angles, normalised_img, arange(0.1,1,0.1),linewidths=1, cmap=plt.cm.coolwarm)    
    plt.clabel(cset, inline=True, fmt='%1.1f', fontsize=20)    
    
    labels = ["x angle / deg", "y angle / deg", "efficiency"]
    ax = fig.gca()       
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.text(0.1, 0.9, description, transform=ax.transAxes, color='w', fontsize=30)
    ax.set_aspect('equal', adjustable='box')

def get_effs(focus_position_many, pdr):
    #get eff for a 2d array for focus positions (so 3d array input)
    shp = focus_position_many.shape[0:2]
    aols = [set_up_aol(op_wavelength, focus_position=f, base_freq=base_freq, pair_deflection_ratio=pdr) for f in focus_position_many.reshape(prod(shp), 3)] 
    effs = [calculate_efficiency(a) for a in aols]
    return array(effs).reshape(shp)
    
def calculate_efficiency(aol):
    time_array = (arange(3)-1)*2e-6
    energy = 0
    ray_count = 0

    for time in time_array:
        rays = get_ray_bundle(op_wavelength)
        (_,energies) = aol.propagate_to_distance_past_aol(rays, time, 0)
        energy += sum(energies[:,-1])
        ray_count += len(rays)
                
    return power(energy / ray_count, 2)
    
if __name__ == '__main__':
    plot_fov_surf(1e9, 0)
    
