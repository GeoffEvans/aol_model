from numpy import meshgrid, vectorize
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D

rcParams.update({'lines.linewidth': 3})
rcParams.update({'font.size': 20})
rcParams['svg.fonttype'] = 'none' # No text as paths. Assume font installed.
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.sans-serif'] = ['Arial']
rcParams['font.family'] = 'sans-serif'
rcParams.update({'figure.autolayout': True})


def generic_plot_surface(x_array, y_array, z_func, labels):
    (x, y) = meshgrid(x_array, y_array)
    z_func_vec = vectorize(z_func)
    z = z_func_vec(x, y)
    generic_plot_surface_vals(x, y, z, labels)

def generic_plot_surface_vals(x, y, z, labels):
    fig = plt.figure()
    ax = fig.gca()
    cs = ax.pcolor(x, y, z, cmap='bone')
    cs.set_rasterized(True)
    def onclick(event):
        print '[%f, %f]' % (event.xdata, event.ydata)
    fig.canvas.mpl_connect('button_press_event', onclick)

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    cb = plt.colorbar(cs, orientation = 'vertical')
    cb.set_label(labels[2])
    cb.solids.set_rasterized(True)
    #cs.set_clim(vmin=0,vmax=1)
    plt.tick_params(direction='out')
    plt.show()

def generic_plot_vals(x, y, labels, limits=0):
    plt.plot(x, y)#, 'g:')
    plt.tick_params(direction='out')
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    if not limits == 0:
        plt.axis(limits)
    plt.show()

def generic_plot(x, y_func, labels, limits=0):
    y_func_vec = vectorize(y_func)
    y = y_func_vec(x)
    generic_plot_vals(x, y, labels, limits)

def multi_line_plot(x, y_func_many, labels, lgnd, limits=0):
    y_many = []
    for y_func in y_func_many:
        y_func_vec = vectorize(y_func)
        y_many.append(y_func_vec(x))

    multi_line_plot_vals(x, y_many, labels, lgnd, limits)

def multi_line_plot_vals(x, y_many, labels, lgnd, limits=0):
    for y in y_many:
        plt.plot(x, y)

    plt.legend(lgnd, loc=2)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    if not limits == 0:
        plt.axis(limits)
    plt.grid()
    plt.show()