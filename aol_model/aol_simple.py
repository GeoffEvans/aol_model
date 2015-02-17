from numpy import array, dtype, pi, concatenate, zeros, append
from acoustics import AcousticDrive
from aol_drive import get_reduced_spacings, calculate_drive_freq_4
from error_utils import check_is_unit_vector, check_is_of_length, check_is_singleton
import copy

# Simplified AOL that treats AODs as thin and uses lambda*F/V to calculate deflection angles. 
# Can work with ray or ray_paraxial
class AolSimple(object):

    @staticmethod
    def create_aol(order, op_wavelength, ac_velocity, aod_spacing, base_freq, pair_deflection_ratio, focus_position, focus_velocity, crystal_thickness=[0]*4):
        # useful for converting a 'real' aol into the simple aol
        reduced_spacing = get_reduced_spacings(crystal_thickness[0:3], aod_spacing) # reduce spacings because crystals are taken to be thin
        z_reduced_focus_position = get_reduced_spacings(crystal_thickness[3], focus_position[2])
        reduced_focus_position = concatenate( (focus_position[0:2], [z_reduced_focus_position]) )
        
        (const, linear, _) = calculate_drive_freq_4(order, op_wavelength, ac_velocity, reduced_spacing, [0]*4, base_freq, \
                                                    pair_deflection_ratio, reduced_focus_position, focus_velocity)
        
        check_is_singleton(ac_velocity) # simple drive theory only handles single velocity
        return AolSimple.create_aol_from_drive(order, reduced_spacing, const, linear, op_wavelength)

    @staticmethod
    def create_aol_from_drive(order, aod_spacing, const, linear, op_wavelength):
        acoustic_drives = AcousticDrive.make_acoustic_drives(const, linear)

        aol = AolSimple(order, aod_spacing, acoustic_drives)
        aol.set_base_ray_positions(op_wavelength)
        return aol

    def __init__(self, order, aod_spacing, acoustic_drives, base_ray_positions=zeros((4,2)), aod_directions=[[1,0,0],[0,1,0],[-1,0,0],[0,-1,0]]):
        self.order = order
        self.aod_spacing = array(aod_spacing, dtype=dtype(float))
        self.acoustic_drives = acoustic_drives
        self.aod_directions = array(aod_directions, dtype=dtype(float))
        self.base_ray_positions = array(base_ray_positions, dtype=dtype(float))
        
        for d in self.aod_directions:
            check_is_unit_vector(d)
        check_is_of_length(3, self.aod_spacing)
        check_is_of_length(4, self.acoustic_drives)
        check_is_of_length(4, self.aod_directions)
        check_is_of_length(4, self.base_ray_positions)
    
    def update_drive(self, focus_position, focus_velocity, op_wavelength, base_freq, pair_deflection_ratio, crystal_thickness):
        ac_velocity = [a.acoustic_drives.velocity for a in self.aods]
        focus_position[2] = get_reduced_spacings(crystal_thickness[3], focus_position[2])
        
        (const, linear, _) = calculate_drive_freq_4(self.order, op_wavelength, ac_velocity, self.aod_spacing, [0]*4, \
                                        base_freq, pair_deflection_ratio, focus_position, focus_velocity)
        
        self.acoustic_drives = AcousticDrive.make_acoustic_drives(const, linear) 
        self.set_base_ray_positions(op_wavelength)
        
    def set_base_ray_positions(self, op_wavelength):
        self.base_ray_positions = self.find_base_ray_positions(op_wavelength)
    
    def find_base_ray_positions(self, op_wavelength):
        from ray_paraxial import RayParaxial
        tracer_ray = RayParaxial([0,0,0], [0,0,1], op_wavelength)
        
        linear = [0]*4
        for k in range(4):
            linear[k] = self.acoustic_drives[k].linear # store values
            self.acoustic_drives[k].linear = 0
        
        path = self.propagate_to_distance_past_aol(tracer_ray, 0)

        for k in range(4):
            self.acoustic_drives[k].linear = linear[k] # restore values
        
        return path[:-1,0:2]
    
    def plot_ray_through_aol(self, ray, time, distance):
        import matplotlib as mpl
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        from numpy import meshgrid, atleast_2d
        
        new_ray = copy.deepcopy(ray)
        new_ray.propagate_free_space_z(self.aod_spacing.sum())

        path = self.propagate_to_distance_past_aol(new_ray, time, distance)
        path_extended = concatenate( (atleast_2d(ray.position), path) )
        
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(path_extended[:,0], path_extended[:,1], path_extended[:,2])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')  

        def add_planes():        
            for point in path_extended[1:5]:
                (xpts, ypts) = meshgrid([1., -1.], [1., -1.])
                xpts += point[0]
                ypts += point[1]
                zpts = point[2] + zeros((2,2))
                ax.plot_surface(xpts, ypts, zpts, color='blue', alpha=.3, linewidth=0, zorder=3)

        add_planes()             
        plt.show()
    
    def propagate_to_distance_past_aol(self, ray, time, distance=0):
        spacings = append(self.aod_spacing, distance) 
        path = zeros( (5,3) )
            
        def diffract_and_propagate(aod_num):
            path[aod_num-1,:] = ray.position
            self.diffract_at_aod(ray, time, aod_num)
            ray.propagate_free_space_z(spacings[aod_num-1])

        for k in range(spacings.size):
            diffract_and_propagate(k+1)
        
        path[4,:] = ray.position
        return path
        
    def diffract_at_aod(self, ray, time, aod_number):
        idx = aod_number-1
        
        aod_dir = self.aod_directions[idx]
        drive = self.acoustic_drives[idx]
        
        local_acoustics = drive.get_local_acoustics(time, [ray.position], self.base_ray_positions[idx], aod_dir)[0] # only want singleton until class extended to support many rays
        
        wavevector_shift = self.order * (2 * pi * local_acoustics.frequency / local_acoustics.velocity) * aod_dir 
        ray.wavevector_vac += wavevector_shift 