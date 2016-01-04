import aod_visualisation as a
import numpy as np

op_wavelength_vac = 920e-9
is_wide = True
order=-1
resolution=90
freq_bnds=(20,100)
deg_bnds=(0.5,5.5)

ac_rot_narrow = a.AodVisualisation(op_wavelength_vac, [np.cos(0.0),0,-np.sin(0.0)], False, order, resolution, freq_bnds, (0.5,4.5))
ac_rot_narrow.plot_efficiency_xangle_freq(ac_power=1.5)
ac_rot_0 = a.AodVisualisation(op_wavelength_vac, [np.cos(0.0),0,-np.sin(0.0)], is_wide, order, resolution, freq_bnds, (0.5,4.5))
ac_rot_0.plot_efficiency_xangle_freq(ac_power=1.5)
ac_rot_3 = a.AodVisualisation(op_wavelength_vac, [np.cos(0.05),0,-np.sin(0.05)], is_wide, order, resolution, freq_bnds, (8.5,12.5))
ac_rot_3.plot_efficiency_xangle_freq(ac_power=1.5)
ac_rot_4 = a.AodVisualisation(op_wavelength_vac, [np.cos(0.07),0,-np.sin(0.07)], is_wide, order, resolution, freq_bnds, (11.5,15.5))
ac_rot_4.plot_efficiency_xangle_freq(ac_power=1.5)
ac_rot_4 = a.AodVisualisation(op_wavelength_vac, [np.cos(0.07),0,-np.sin(0.07)], is_wide, order, resolution, freq_bnds, (11.5,15.5))
ac_rot_4.plot_efficiency_xangle_freq(ac_power=20)
ac_rot_8 = a.AodVisualisation(op_wavelength_vac, [np.cos(0.14),0,-np.sin(0.14)], is_wide, order, resolution, freq_bnds, (27.5,31.5))
ac_rot_8.plot_efficiency_xangle_freq(ac_power=1.5)