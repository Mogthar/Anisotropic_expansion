import numpy as np
import scipy.constants

m_err = 167.259 * scipy.constants.atomic_mass

class Potential():

    def value(self, r):
        pass

    def gradient(self, r):
        pass

    def integrationLimit(self, T, tolerance):
        pass


class HarmonicPotential(Potential):

    def __init__(self, omega):
        self.omega = omega

    def value(self, r):
        return 0.5 * m_err * np.sum(self.omega**2 * r**2)
    
    def gradient(self, r):
        return m_err * self.omega**2 * r
    
    def integrationLimit(self, T, tolerance):
        return np.sqrt(2 * scipy.constants.Boltzmann * T * tolerance / m_err) / self.omega
    
