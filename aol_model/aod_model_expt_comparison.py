from aod_visualisation import AodVisualisation
from vector_utils import normalise
import expt_data as d
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

rcParams.update({'lines.linewidth': 3})
rcParams.update({'font.size': 20})
rcParams['svg.fonttype'] = 'none' # No text as paths. Assume font installed.
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.sans-serif'] = ['Arial']
rcParams['font.family'] = 'sans-serif'

av_wide = AodVisualisation(785e-9, ac_dir_rel=[1,0,0], is_wide=True, deg_bnds=(0,4))
av_narrow = AodVisualisation(785e-9, ac_dir_rel=normalise([1,0,0]), is_wide=False, deg_bnds=(-2,6))
av_narrow_800 = AodVisualisation(800e-9, ac_dir_rel=normalise([1,0,0]), is_wide=False, deg_bnds=(-1,6))
av_narrow_909 = AodVisualisation(909e-9, ac_dir_rel=normalise([1,0,0]), is_wide=False, deg_bnds=(-1,6))
av_wide_800 = AodVisualisation(800e-9, ac_dir_rel=normalise([1,0,0]), is_wide=True, deg_bnds=(-1,6))
av_wide_909 = AodVisualisation(909e-9, ac_dir_rel=normalise([1,0,0]), is_wide=True, deg_bnds=(-1,6))

def plot_eff_pwr_wide():
    plt.plot(d.power, d.eff_power_wide, 'o')
    av_wide.plot_efficiency_power()

def plot_eff_pwr_narrow():
    plt.plot(d.power, d.eff_power_narrow, 'o')
    av_narrow.plot_efficiency_power()


def plot_eff_freq_wide():
    plt.plot(d.freq_wide, d.eff_freq_wide, 'o')
    av_wide.plot_efficiency_freq_max()

def plot_eff_freq_narrow():
    plt.plot(d.freq_narrow, d.eff_freq_narrow, 'o')
    av_narrow.plot_efficiency_freq_max()

def plot_eff_ang_wide():
    av_wide.plot_efficiency_xangle(ac_power=1.5)
    plt.plot(d.angle_wide_again, d.eff_angle_wide_again, 'ro', markersize=12)
    plt.xticks([0,2,4])
    plt.yticks([0,0.5,1])

def plot_eff_ang_narrow():
    av_narrow.plot_efficiency_xangle(ac_power=1.5)
    plt.plot(d.angle_narrow_again, d.eff_angle_narrow_again, 'ro', markersize=12)
    plt.xticks([-2,2,6])
    plt.yticks([0,0.5,1])

def plot_eff_freq_narrow_expt_model():
    plt.plot(d.freq_narrow_new, d.eff_freq_narrow_909_1, 'bo', markersize=8)
    plt.plot(d.freq_narrow_new, d.eff_freq_narrow_800_1, 'go', markersize=8)
    plt.plot(d.freq_narrow_new, d.eff_freq_narrow_909_2, 'ro', markersize=8)
    plt.plot(d.freq_narrow_new, d.eff_freq_narrow_800_23, 'co', markersize=8)

    av_narrow_909.plot_efficiency_freq_max()
    av_narrow_800.plot_efficiency_freq_max()
    av_narrow_909.plot_efficiency_freq_max_second_order()
    av_narrow_800.plot_efficiency_freq_max_second_order()

    plot_narrow_transducer_eff()
    label_list = ['909nm -1 mode expt', '800nm -1 mode expt', '909nm -2 mode expt', '800nm -2 mode expt', \
                  '909nm -1 mode model', '800nm -1 mode model', '909nm -2 mode model', '800nm -2 mode model', \
                  'RF to acoustic inferred', 'RF to acoustic spline']
    plt.legend(label_list, loc=0, borderaxespad=1.6, fontsize=16)

def plot_eff_freq_wide_expt_model():
    av_wide_909.plot_efficiency_freq_max()
    av_wide_800.plot_efficiency_freq_max()
    av_wide_909.plot_efficiency_freq_max_second_order()
    av_wide_800.plot_efficiency_freq_max_second_order()

    plt.plot(d.freq_wide_new, d.eff_freq_wide_909_1, 'bo', markersize=12)
    plt.plot(d.freq_wide_new, d.eff_freq_wide_800_1, 'go', markersize=12)
    plt.plot(d.freq_wide_new, d.eff_freq_wide_909_2, 'ro', markersize=12)
    plt.plot(d.freq_wide_new, d.eff_freq_wide_800_2, 'co', markersize=12)

    plot_wide_transducer_eff()
    label_list = ['909nm -1 mode expt', '800nm -1 mode expt', '909nm -2 mode expt', '800nm -2 mode expt', \
                  '909nm -1 mode model', '800nm -1 mode model', '909nm -2 mode model', '800nm -2 mode model', \
                  'RF to acoustic inferred', 'RF to acoustic spline']
    plt.legend(label_list, loc=0, borderaxespad=1.6, fontsize=16)

def plot_wide_transducer_eff():
    from aol_model.set_up_utils import transducer_efficiency_wide
    f = np.linspace(20, 50, 300) * 1e6
    plt.plot(d.freq_wide_new, transducer_efficiency_wide(np.array(d.freq_wide_new)*1e6), 'mo')
    plt.plot(f/1e6, transducer_efficiency_wide(f))

def plot_narrow_transducer_eff():
    from aol_model.set_up_utils import transducer_efficiency_narrow
    f = np.linspace(20, 50, 300) * 1e6
    plt.plot(d.freq_narrow_new, transducer_efficiency_narrow(np.array(d.freq_narrow_new)*1e6), 'mo', markersize=8)
    plt.plot(f/1e6, transducer_efficiency_narrow(f))

def plot_transducer_eff():
    plot_narrow_transducer_eff()
    plot_wide_transducer_eff()
    plt.plot(d.freq_narrow_new, d.fwd_ref_eff_narrow, marker='o')
    plt.plot(d.freq_wide_new, d.fwd_ref_eff_wide, marker='o')

if __name__ == '__main__':
    #plot_transducer_eff()

    #plt.figure()
    #plot_eff_freq_narrow_expt_model()
    #plt.figure()
    #plot_eff_freq_wide_expt_model()

    #plt.figure()
    #plot_eff_freq_narrow()
    #plt.figure()
    #plot_eff_freq_wide()
    #plt.figure()
    plot_eff_ang_wide()
    #plt.figure()
    plot_eff_ang_narrow()
    #plt.figure()
    #plot_eff_pwr_narrow()
    #plt.figure()
    #plot_eff_pwr_wide()
