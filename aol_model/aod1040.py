from aod_visualisation import AodVisualisation
import numpy as np
import matplotlib.pyplot as plt

av_narrow = AodVisualisation(1040e-9, ac_dir_rel=[1,0,0], is_wide=False, deg_bnds=(-1,6), freq_bnds=(20,60))
av_narrow.aod.transducer_width = 1.2e-3
av_wide = AodVisualisation(1040e-9, ac_dir_rel=[1,0,0], is_wide=True, deg_bnds=(-1,6), freq_bnds=(25,55))
av_wide.aod.transducer_width = 3.8e-3
av_narrow800 = AodVisualisation(800e-9, ac_dir_rel=[1,0,0], is_wide=False, deg_bnds=(-1,6), freq_bnds=(20,60))
av_narrow800.aod.transducer_width = 1.2e-3
av_wide800 = AodVisualisation(800e-9, ac_dir_rel=[1,0,0], is_wide=True, deg_bnds=(-1,6), freq_bnds=(25,55))
av_wide800.aod.transducer_width = 3.8e-3

f = np.linspace(30,50,21)
e1040 = np.array([25, 35, 37, 43, 53, 62, 70, 78, 83.4, 88, 87, 83, 74, 62, 47, 31, 18, 9, 4.5, 4.8, 2]) / 100
e800 = np.array([57, 69, 71, 73, 77, 83, 85.5, 86.5, 87.5, 88, 86, 84, 79, 73, 66, 56, 47, 36, 25, 16, 8]) / 100

#fn = np.linspace(25,55,31)
#n800 = np.array([40. ,33 ,27 ,24 ,23 ,23 ,24 ,32 ,40 ,50 ,60 ,69 ,76 ,77 ,78 ,77 ,76 ,77 ,78 ,81 ,83 ,83 ,81 ,79 ,76 ,73 ,69 ,63 ,54 ,41 ,29]) / 100
#n1040 = np.array([38 ,42 ,50 ,59 ,67 ,73 ,72 ,72 ,71 ,74 ,76 ,78 ,78 ,79 ,78 ,76 ,73 ,69 ,66 ,62 ,56 ,49 ,43 ,37 ,30 ,22 ,16 ,11 ,7.5 ,4.0 ,2.6]) / 100

av_wide.plot_efficiency_angle_out(ac_power=1.8, deg=2.24)
av_wide800.plot_efficiency_angle_out(ac_power=1., deg=2.03)
#av_narrow.plot_efficiency_freq(ac_power=7, deg=2.3)
#av_narrow800.plot_efficiency_freq(ac_power=4.5, deg=2.1)

plt.plot((f-39)*1.04/613, e1040, 'bo')
plt.plot((f-39)*0.8/613, e800, 'go')
#plt.plot(fn, n1040, 'bx')
#plt.plot(fn, n800, 'gx')