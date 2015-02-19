"""The acoustics module provides the means to handle the acoustic wave involved
in the acousto-optic interaction. An AOL associates an AcousticDrive object with
each AOD object. When an optic ray (Ray object) is incident on the AOD, the
AcousticDrive is called to generate the local acoustics, an Acoustics object.
The Acoustics object is used in the xu_stroud_model module to handle the
acousto-optic interaction."""

from numpy import pi, dot, dtype, array, sqrt
import numpy as np

teo2_ac_vel = 612.8834
pointing_ramp_time = 30e6
default_power = 1

class Acoustics(object):
    """A class to hold the acoustic properties relevant to the acousto-optic
    interaction. An Acoustics object is generated from an AcousticDrive object
    for each Ray incident on an AOD object. The Acoustics object is used in
    the xu_stroud_model module to handle the acousto-optic interaction."""

    def __init__(self, frequency, power=default_power, velocity=teo2_ac_vel):
        self.frequency = frequency
        self.power = power
        self.velocity = velocity

    @property
    def wavevector_mag(self):
        return 2 * pi * self.frequency / self.velocity

    def wavevector(self, aod):
        return self.wavevector_mag * aod.acoustic_direction

    def amplitude(self, aod):
        teo2_density = 5990;
        numerator = 2 * self.power
        denominator = teo2_density * self.velocity**3 * aod.transducer_width * aod.transducer_height # use of aperture width assumes square aperture
        return sqrt(numerator / denominator) # Xu & Stroud (2.143)

class AcousticDrive(object):
    """A class to hold the drive parameters for an AOD. An AOL associates an
    AcousticDrive object with each of its AODs. When an optic ray (Ray object)
    is incident on the AOD, the AcousticDrive is called to generate the local
    acoustics, an Acoustics object."""

    @staticmethod
    def make_acoustic_drives(const, linear, quad=[0]*4, power=[default_power]*4, velocity=teo2_ac_vel, ramp_time=None):
        """Generates four AcousticDrive objects at a time. The function is useful
        because and AOL has four AODs, and each AOD requires a corresponding AcousticDrive."""
        acoustic_drives = [0]*4
        for k in range(4):
            acoustic_drives[k] = AcousticDrive(const[k], linear[k], quad[k], power[k], velocity, ramp_time)
        return array(acoustic_drives)

    def __init__(self, const, linear, quad=0, power=default_power, velocity=teo2_ac_vel, ramp_time=None):
        self.const = array(const, dtype=dtype(float))
        self.linear = array(linear, dtype=dtype(float))
        self.quad = array(quad, dtype=dtype(float))
        self.power = array(power, dtype=dtype(float))
        self.velocity = velocity
        self.ramp_time = ramp_time

    def get_local_acoustics(self, time, ray_positions, base_ray_position, aod_direction):
        """Returns an Acoustics object to represent the local acoustic field at the
        point an optic ray is incident on an AOD."""
        distances = dot(array(ray_positions)[:,0:2] - base_ray_position, aod_direction[0:2])
        effective_time = time - distances/self.velocity

        if self.ramp_time is not None:
            t = effective_time - np.floor(effective_time/self.ramp_time + 0.5) * self.ramp_time
        else: # if there is no ramp time, there is no ramp
            t = effective_time

        frequencies = self.const + self.linear * t + self.quad * np.power(t, 2.)
        return [Acoustics(f, self.power, self.velocity) for f in frequencies]