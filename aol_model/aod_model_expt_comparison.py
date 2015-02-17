from aod_visualisation import AodVisualisation
from aol_model.vector_utils import normalise
import expt_data as d
import matplotlib.pyplot as plt

av_wide = AodVisualisation(800e-9, ac_dir_rel=[1,0,0], is_wide=True, deg_bnds=(-1,4))
av_narrow = AodVisualisation(800e-9, ac_dir_rel=normalise([1,0,0]), is_wide=False, deg_bnds=(-1,6))

def plot_eff_pwr_wide():
    plt.plot(d.power, d.eff_power_wide, 'rx')
    av_wide.plot_efficiency_power()
    
def plot_eff_freq_wide():
    plt.plot(d.freq_wide, d.eff_freq_wide, 'rx')
    av_wide.plot_efficiency_freq_max()
    
def plot_eff_ang_wide():
    plt.plot(d.angle_wide, d.eff_angle_wide, 'rx')
    av_wide.plot_efficiency_xangle()

def plot_eff_pwr_narrow():
    plt.plot(d.power, d.eff_power_narrow, 'kx')
    av_narrow.plot_efficiency_power() 

def plot_eff_freq_narrow():
    plt.plot(d.freq_narrow, d.eff_freq_narrow, 'kx')
    av_narrow.plot_efficiency_freq_max()
    
def plot_eff_ang_narrow():
    plt.plot(d.angle_narrow, d.eff_angle_narrow, 'kx')
    av_narrow.plot_efficiency_xangle()
    
if __name__ == '__main__':
    plot_eff_freq_narrow()
    plot_eff_freq_wide()
    #plot_eff_ang_narrow()
    #plot_eff_ang_wide()
    #plot_eff_pwr_narrow()    
    #plot_eff_pwr_wide()
