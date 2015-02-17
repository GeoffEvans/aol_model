from aol_simple import AolSimple
from acoustics import AcousticDrive, default_power, teo2_ac_vel
from aol_drive import calculate_drive_freq_4
from acoustics import pointing_ramp_time
from numpy import append, array, dtype, concatenate, zeros, atleast_2d, dot,\
    isnan
import copy

# AOL model using AOD objects, incoroporating the Xu & Stroud diffraction theory.
class AolFull(object):
       
    @staticmethod
    def create_aol(aods, aod_spacing, order, op_wavelength, base_freq, pair_deflection_ratio, focus_position, \
            focus_velocity, ac_power=[default_power]*4, ac_velocity=teo2_ac_vel, ramp_time=pointing_ramp_time):

        crystal_thickness = array([a.crystal_thickness for a in aods], dtype=dtype(float))
        (const, linear, quad) = calculate_drive_freq_4(order, op_wavelength, ac_velocity, aod_spacing, crystal_thickness, \
                                base_freq, pair_deflection_ratio, focus_position, focus_velocity)
       
        acoustic_drives = AcousticDrive.make_acoustic_drives(const, linear, quad, ac_power, ac_velocity, ramp_time)
        return AolFull(aods, aod_spacing, acoustic_drives, order, op_wavelength)
       
    def __init__(self, aods, aod_spacing, acoustic_drives, order, op_wavelength): 
        self.aods = array(aods)
        self.aod_spacing = array(aod_spacing, dtype=dtype(float))
        self.acoustic_drives = array(acoustic_drives)
        self.order = order
        
        simple = AolSimple(order, self.aod_spacing, self.acoustic_drives)
        self.base_ray_positions = simple.find_base_ray_positions(op_wavelength)
    
    def plot_ray_through_aol(self, rays, time, distance):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from numpy import meshgrid, atleast_3d, mean
        
        num_rays = len(rays)
        new_rays = [0]*num_rays
        for m in range(num_rays): 
            new_rays[m] = copy.deepcopy(rays[m]) # don't want to alter the ray state
            new_rays[m].propagate_free_space_z(self.aod_spacing.sum())

        (paths, _) = self.propagate_to_distance_past_aol(new_rays, time, distance)
        start = atleast_3d([r.position for r in rays]).transpose((0,2,1))
        paths_extended = concatenate((start, paths), axis=1)
        
        ax = plt.gca(projection='3d')
        for m in range(num_rays):
            ax.plot(paths_extended[m,:,0], paths_extended[m,:,1], paths_extended[m,:,2])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')  

        def add_planes():        
            pts = mean(paths_extended[:,1:9,:], axis=0)
            for m in range(8):
                normal = self.aods[m/2].normal # integer division
                axis_z = dot(pts[m], normal) / normal[2] # value of z plane cuts on axis
                (xpts, ypts) = array(meshgrid([1, -1], [1, -1])) * 1e-2
                xpts += pts[m][0]
                ypts += pts[m][1]
                zpts = axis_z - (xpts * normal[0] + ypts * normal[1]) / normal[2]
                ax.plot_surface(xpts, ypts, zpts, color='blue', alpha=.3, linewidth=0, zorder=3)

        add_planes()             
        plt.show()
        return plt
    
    def propagate_to_distance_past_aol(self, rays, time, distance=0):
        num_rays = len(rays)
        crystal_thickness = array([a.crystal_thickness for a in self.aods], dtype=dtype(float))
        spacings = append(self.aod_spacing, distance) 
        normals = concatenate( ([a.normal for a in self.aods], atleast_2d([0,0,1])) )
        paths = zeros( (len(rays),9,3) )
        energies = zeros( (len(rays),4) )

        for m in range(num_rays): # move rays to entrance of first crystal
                rays[m].propagate_from_plane_to_plane(0, array([0.,0.,1.]), self.aods[0].normal)
        
        def diffract_and_propagate(aod_num):
            for m in range(num_rays):   
                paths[m,2*aod_num - 2,:] = rays[m].position     # set path at entrance
            self.diffract_at_aod(rays, time, aod_num)           # diffract at crystal
            for m in range(num_rays):
                paths[m,2*aod_num - 1,:] = rays[m].position     # set path at exit
                energies[m,aod_num-1] = rays[m].energy          # (line below) move ray to entrance of next crystal
                reduced_spacing = spacings[aod_num-1] - crystal_thickness[aod_num-1]/dot(self.aods[aod_num-1].normal, array([0,0,1]))
                rays[m].propagate_from_plane_to_plane(reduced_spacing, normals[aod_num-1], normals[aod_num])        
        
        for k in range(4):
            diffract_and_propagate(k+1)
        
        for m in range(num_rays):
            paths[m,8,:] = rays[m].position
        return (paths, energies)
    
    def diffract_at_aod(self, rays, time, aod_number):
        idx = aod_number-1
        
        aod = self.aods[idx]
        base_ray_position = self.base_ray_positions[idx]
        drive = self.acoustic_drives[idx]
        local_acoustics = drive.get_local_acoustics(time, [r.position for r in rays], base_ray_position, aod.acoustic_direction)
        
        aod.propagate_ray(rays, local_acoustics, self.order)
        
    def change_orientation(self, aod_num, new_normal):
        assert not any(isnan(new_normal))
        self.aods[aod_num-1].normal = array(new_normal)
        
if __name__ == "__main__":
    import aol_model.set_up_utils as s
    rays = s.get_ray_bundle(800e-9)
    aol = s.set_up_aol(800e-9)    
    aol.plot_ray_through_aol(rays, 0, 1)