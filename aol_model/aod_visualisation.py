from ray import Ray
from acoustics import Acoustics
from numpy import linspace, pi, sin, cos, abs, sqrt, arcsin, max, array, argmax
from plot_utils import generic_plot_surface, generic_plot, multi_line_plot
from xu_stroud_model import diffract_by_wavevector_triangle
from set_up_utils import make_aod_narrow, make_aod_wide

class AodVisualisation(object):
    """Useful class for exploring AOD properties graphically."""
    def __init__(self, op_wavelength_vac, \
            ac_dir_rel=[1,0,0], \
            is_wide=True, \
            order=-1, \
            resolution=90, \
            freq_bnds=(20,60), \
            deg_bnds=(0.5,3.5), \
            ):
        normal = [0,0,1]
        self.aod = make_aod_narrow(normal, ac_dir_rel)
        if is_wide:
            self.aod = make_aod_wide(normal, ac_dir_rel)
        self.order = order
        self.op_wavelength_vac = op_wavelength_vac
        self.resolution = resolution
        self.mhz_range = linspace(freq_bnds[0], freq_bnds[1], resolution)
        self.degrees_range =  linspace(deg_bnds[0], deg_bnds[1], resolution)

    def plot_mismatch_angle_freq(self, ac_power=1.5):
        """Plot wavevector mismatch between incident optic wavevector,
        acoustic wavevector and diffracted optic wavevector against
        incidence angle and acoustic frequency."""
        def func(deg, mhz):
            ang = deg * pi/180
            wavevector_unit = [sin(ang), 0, cos(ang)]
            ray = Ray([0,0,0], wavevector_unit, self.op_wavelength_vac)
            acoustics = Acoustics(mhz*1e6, ac_power)

            (mismatch,_) = diffract_by_wavevector_triangle(self.aod, [ray], [acoustics], self.order, (0,1))
            return abs(mismatch)

        labels = ["Incidence angle / deg","frequency / MHz","Wavevector mismatch / 1/m"]
        generic_plot_surface(self.degrees_range, self.mhz_range, func, labels)

    def plot_efficiency_xangle_freq(self, ac_power=1.5):
        """Plot diffraction efficiency against
        incidence angle and acoustic frequency."""
        def func(deg, mhz):
            ang = deg * pi/180
            optical_rot = pi/180 * 0
            wavevector_unit = [cos(optical_rot)*sin(ang), sin(optical_rot), cos(optical_rot)*cos(ang)]
            ray = Ray([0,0,0], wavevector_unit, self.op_wavelength_vac)
            acoustics = Acoustics(mhz*1e6, ac_power)

            self.aod.propagate_ray([ray], [acoustics], self.order)
            return ray.energy

        labels = ["Incidence angle / deg","Frequency / MHz","Efficiency"]
        generic_plot_surface(self.degrees_range, self.mhz_range, func, labels)

    def plot_efficiency_xangle_freq_second_order_noise(self, ac_power=1.5):
        """Plot diffraction efficiency against
        incidence angle and acoustic frequency."""
        def func(deg, mhz):
            ang = deg * pi/180
            wavevector_unit = [sin(ang), 0, cos(ang)]
            ray = Ray([0,0,0], wavevector_unit, self.op_wavelength_vac)
            acoustics = Acoustics(mhz*1e6, ac_power)

            self.aod.propagate_ray([ray], [acoustics], self.order)
            return ray.resc / (ray.energy + 1e-6)

        labels = ["Incidence angle / deg","Frequency / MHz","Efficiency"]
        generic_plot_surface(self.degrees_range, self.mhz_range, func, labels)

    def plot_efficiency_xangle_yangle(self, freq, ac_power=1.5):
        """Plot diffraction efficiency against
        incidence angle and the orthogonal incidence angle."""
        def func(deg, deg_trans):
            ang = deg * pi/180
            ang_trans = deg_trans * pi/180
            wavevector_unit = [ang, ang_trans, sqrt(1 - ang**2 - ang_trans**2)]
            ray = Ray([0,0,0], wavevector_unit, self.op_wavelength_vac)
            acoustics = Acoustics(freq, ac_power)

            self.aod.propagate_ray([ray], [acoustics], self.order)
            return ray.energy

        labels = ["Incidence angle / deg","Transverse incidence angle / deg","Efficiency"]
        generic_plot_surface(linspace(0.03, 0.1, 30)/pi*180, linspace(-0.3, 0.3, 60)/pi*180, func, labels)

    def plot_xangleout_xangle_freq(self, ac_power=1.5):
        """Plot diffracted angle against
        incidence angle and the acoustic frequency."""
        def func(deg, mhz):
            ang = deg * pi/180
            wavevector_unit = [sin(ang), 0, cos(ang)]
            ray = Ray([0,0,0], wavevector_unit, self.op_wavelength_vac)
            acoustics = Acoustics(mhz*1e6, ac_power)

            self.aod.propagate_ray([ray], [acoustics], self.order)
            return arcsin(ray.wavevector_unit[0]) * 180/pi

        labels = ["Incidence angle / deg","Frequency / MHz","Diffracted angle / deg"]
        generic_plot_surface(self.degrees_range, self.mhz_range, func, labels)

    def plot_xangleout_xangle_yangle(self, ac_power=1.5):
        """Plot diffracted angle against
        incidence angle and the orthogonal incidence angle."""
        def func(deg, deg_trans):
            ang = deg * pi/180
            ang_trans = deg_trans * pi/180
            wavevector_unit = [sin(ang), sin(ang_trans), sqrt(1 - sin(ang)**2 - sin(ang_trans)**2)]
            ray = Ray([0,0,0], wavevector_unit, self.op_wavelength_vac)
            acoustics = Acoustics(35e6, ac_power)

            self.aod.propagate_ray([ray], [acoustics], self.order)
            return arcsin(ray.wavevector_unit[0]) * 180/pi

        labels = ["Incidence angle / deg","Transverse incidence angle / deg","Diffracted angle / deg"]
        generic_plot_surface(self.degrees_range, linspace(-5, 5, self.resolution), func, labels)

    def plot_efficiency_freq(self, ac_power=1.5, deg=2.2):
        """Plot diffraction efficiency against acoustic frequency for fixed incidence angle."""
        def func(mhz):
            ang = deg * pi / 180
            wavevector_unit = [sin(ang), 0, cos(ang)]
            ray = Ray([0,0,0], wavevector_unit, self.op_wavelength_vac)
            acoustics = Acoustics(mhz*1e6, ac_power)

            self.aod.propagate_ray([ray], [acoustics], self.order)
            return ray.energy

        labels = ["Frequency / MHz","Efficiency"]
        generic_plot(self.mhz_range, func, labels)

    def plot_efficiency_freq_max(self, ac_power=1.5):
        """Plot maximum diffraction efficiency for any incidence angle against acoustic frequency."""
        def func(mhz):
            deg_range =  linspace(0.9, 3, 70)
            rad_range = deg_range * pi / 180
            rays = [Ray([0,0,0], [sin(ang), 0, cos(ang)], self.op_wavelength_vac) for ang in rad_range]
            acoustics = Acoustics(mhz*1e6, ac_power)

            self.aod.propagate_ray(rays, [acoustics]*len(rays), self.order)
            idx = argmax([r.energy for r in rays])
            return rays[idx].energy

        labels = ["Frequency / MHz","Efficiency"]
        generic_plot(self.mhz_range, func, labels, (min(self.mhz_range),max(self.mhz_range),0,1))

    def plot_efficiency_freq_max_second_order(self, ac_power=1.5):
        """Plot maximum diffraction efficiency for any incidence angle against acoustic frequency."""
        def func(mhz):
            deg_range =  linspace(0.9, 3, 70)
            rad_range = deg_range * pi / 180
            rays = [Ray([0,0,0], [sin(ang), 0, cos(ang)], self.op_wavelength_vac) for ang in rad_range]
            acoustics = Acoustics(mhz*1e6, ac_power)

            self.aod.propagate_ray(rays, [acoustics]*len(rays), self.order)
            idx = argmax([r.energy for r in rays])
            return rays[idx].resc

        labels = ["Frequency / MHz","Efficiency"]
        generic_plot(self.mhz_range, func, labels, (min(self.mhz_range),max(self.mhz_range),0,1))

    def plot_efficiency_freq_max__power_varies_with_wavelength(self, deg=1.95, power_wavelen=[(1.5, 800e-9), (1.5, 920e-9), (1.5, 1030e-9), (2.2, 920e-9)]):
        """Plot maximum diffraction efficiency against acoustic power for fixed acoustic frequency and incidence angle."""
        def create_efficiency_function_closure(ac_power, wavelen):
            def func(mhz):
                deg_range =  array([deg])
                rad_range = deg_range * pi / 180
                rays = [Ray([0,0,0], [sin(ang), 0, cos(ang)], wavelen) for ang in rad_range]
                acoustics = Acoustics(mhz*1e6, ac_power)

                self.aod.propagate_ray(rays, [acoustics]*len(rays), self.order)
                return max([r.energy for r in rays])
            return func

        funcs = []
        for elem in power_wavelen:
            funcs.append(create_efficiency_function_closure(elem[0], elem[1]))

        labels = ["Frequency / MHz","Efficiency"]
        lgnd = power_wavelen
        multi_line_plot(self.mhz_range, funcs, labels, lgnd, (min(self.mhz_range),max(self.mhz_range),0,1))

    def plot_efficiency_xangle(self, ac_power=1.5, ac_mhz=39):
        """Plot diffraction efficiency against incidence angle for fixed acoustic frequency."""
        def func(deg):
            ang = deg * pi/180
            wavevector_unit = [sin(ang), 0, cos(ang)]
            ray = Ray([0,0,0], wavevector_unit, self.op_wavelength_vac)
            acoustics = Acoustics(ac_mhz*1e6, ac_power)

            self.aod.propagate_ray([ray], [acoustics], self.order)
            return ray.energy

        labels = ["Incidence angle / deg","Efficiency"]
        generic_plot(self.degrees_range, func, labels, (min(self.degrees_range),max(self.degrees_range),0,1))

    def plot_efficiency_power(self, ac_mhz=40):
        """Plot maximum diffraction efficiency for any incidence angle against acoustic power for fixed acoustic frequency."""
        def func(ac_power):
            deg_range =  linspace(1.9, 3.1, 40)
            rad_range = deg_range * pi / 180
            rays = [Ray([0,0,0], [sin(ang), 0, cos(ang)], self.op_wavelength_vac) for ang in rad_range]
            acoustics = Acoustics(ac_mhz*1e6, ac_power)

            self.aod.propagate_ray(rays, [acoustics]*len(rays), self.order)
            return max([r.energy for r in rays])

        labels = ["Acoustic power / W","Efficiency"]
        ac_power_range = linspace(0,2.5,20)
        generic_plot(ac_power_range, func, labels, (min(ac_power_range),max(ac_power_range),0,1))

if __name__ == '__main__':
    av = AodVisualisation(700e-9, is_wide=False)
    av.plot_efficiency_xangle_freq(ac_power=1.5)
    av.plot_efficiency_xangle_freq_second_order_noise()
    av = AodVisualisation(700e-9, is_wide=True)
    av.plot_efficiency_xangle_freq(ac_power=1.5)
    #av.plot_efficiency_freq_max()
