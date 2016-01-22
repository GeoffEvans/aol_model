from aod_visualisation import AodVisualisation

av_narrow = AodVisualisation(1040e-9, ac_dir_rel=[1,0,0], is_wide=False, deg_bnds=(-1,6), freq_bnds=(20,60))
av_wide = AodVisualisation(1040e-9, ac_dir_rel=[1,0,0], is_wide=True, deg_bnds=(-1,6), freq_bnds=(20,60))
av_narrow2 = AodVisualisation(800e-9, ac_dir_rel=[1,0,0], is_wide=False, deg_bnds=(-1,6), freq_bnds=(20,60))
av_wide2 = AodVisualisation(800e-9, ac_dir_rel=[1,0,0], is_wide=True, deg_bnds=(-1,6), freq_bnds=(20,60))
#av_narrow.plot_efficiency_xangle_freq(ac_power=1.5)
#av_wide.plot_efficiency_xangle_freq(ac_power=1.5)
#av_narrow.plot_efficiency_freq_max()
#av_wide.plot_efficiency_freq_max()

av_wide.plot_efficiency_freq(ac_power=2.1, deg=2.25)
av_wide2.plot_efficiency_freq(ac_power=1.2, deg=2.05)
av_narrow.plot_efficiency_freq(ac_power=2.1, deg=2.25)
av_narrow2.plot_efficiency_freq(ac_power=1.2, deg=2.05)