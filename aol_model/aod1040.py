from aod_visualisation import AodVisualisation
import numpy as np
import matplotlib.pyplot as plt

av_narrow = AodVisualisation(1040e-9, ac_dir_rel=[1,0,0], is_wide=False, deg_bnds=(-1,6), freq_bnds=(20,60))
av_wide = AodVisualisation(1040e-9, ac_dir_rel=[1,0,0], is_wide=True, deg_bnds=(-1,6), freq_bnds=(25,55))
av_wide.aod.transducer_width = 3.8e-3
av_narrow2 = AodVisualisation(800e-9, ac_dir_rel=[1,0,0], is_wide=False, deg_bnds=(-1,6), freq_bnds=(20,60))
av_wide2 = AodVisualisation(800e-9, ac_dir_rel=[1,0,0], is_wide=True, deg_bnds=(-1,6), freq_bnds=(25,55))
av_wide2.aod.transducer_width = 3.8e-3
#av_narrow.plot_efficiency_xangle_freq(ac_power=1.5)
#av_wide.plot_efficiency_xangle_freq(ac_power=1.5)
#av_narrow.plot_efficiency_freq_max()
#av_wide.plot_efficiency_freq_max()

f = np.linspace(30,50,21)
e1040 = np.array([25, 35, 37, 43, 53, 62, 70, 78, 83.4, 88, 87, 83, 74, 62, 47, 31, 18, 9, 4.5, 4.8, 2]) / 100
e800 = np.array([57, 69, 71, 73, 77, 83, 85.5, 86.5, 87.5, 88, 86, 84, 79, 73, 66, 56, 47, 36, 25, 16, 8]) / 100

av_wide.plot_efficiency_freq(ac_power=1.8, deg=2.24)
av_wide2.plot_efficiency_freq(ac_power=1., deg=2.03)
#av_narrow.plot_efficiency_freq(ac_power=1.8, deg=2.25)
#av_narrow2.plot_efficiency_freq(ac_power=1., deg=2.03)

plt.plot(f, e1040, 'bo')
plt.plot(f, e800, 'go')