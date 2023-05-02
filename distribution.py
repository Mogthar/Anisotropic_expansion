import numpy as np
import scipy.constants
import mpmath
import scipy.integrate as integrate

m_err = 167.259 * scipy.constants.atomic_mass

class Distribution():

    def value(self, p, r, T, mu, Potential):
        pass

    def momentumIntegral(self, r, T, mu, Potential):
        pass
    
    # calculates the average of a COMPONENT of the momentum squared
    def momentumSquaredIntegral(self, r, T, mu, Potential):
        pass

    def norm(self, T, mu, Potential):
        pass

    def averageMomentumSquared(self, T, mu, Potential):
        pass

class MaxwellBoltzmann(Distribution):

    def value(self, p, r, T, mu, Potential):
        kT = scipy.constants.Boltzmann * T
        fugacity = np.exp(mu / (kT))
        return fugacity * np.exp(-Potential.value(r) / kT - np.sum(p**2) / (2 * m_err * kT))
    
    def momentumIntegral(self, r, T, mu, Potential):
        kT = scipy.constants.Boltzmann * T
        return np.exp((mu - Potential.value(r)) / kT) * np.power(2 * np.pi * m_err * kT / scipy.constants.h**2, 3/2)
    
    def momentumSquaredIntegral(self, r, T, mu, Potential):
        kT = scipy.constants.Boltzmann * T
        return np.exp((mu - Potential.value(r)) / kT) * np.power(2 * np.pi * m_err * kT / scipy.constants.h**2, 3/2) * m_err * kT
    
    def norm(self, T, mu, Potential):
        kT = scipy.constants.Boltzmann * T
        fugacity = np.exp(mu / (kT))
        volumeIntegral = integrate.nquad(lambda rx, ry, rz : Potential.value(np.array([rx, ry, rz])), [[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf]])
        return fugacity * np.power(2 * np.pi * m_err * kT / scipy.constants.h**2, 3/2) * volumeIntegral
    
    def averageMomentumSquared(self, T, mu, Potential):
        return m_err * scipy.constants.Boltzmann * T

class BoseEinstein(Distribution):
    
    def value(self, p, r, T, mu, Potential):
        kT = scipy.constants.Boltzmann * T
        fugacity = np.exp(mu / (kT))
        return 1 / (np.exp(Potential.value(r) / kT + np.sum(p**2) / (2 * m_err * kT)) / fugacity - 1)
    
    def momentumIntegral(self, r, T, mu, Potential):
        kT = scipy.constants.Boltzmann * T
        z = np.exp((mu - Potential.value(r))/ (kT))
        return np.power(2 * np.pi * m_err * kT / scipy.constants.h**2, 3/2) * mpmath.polylog(3/2, z)
    
    def momentumSquaredIntegral(self, r, T, mu, Potential):
        kT = scipy.constants.Boltzmann * T
        z = np.exp((mu - Potential.value(r))/ (kT))
        return np.power(2 * np.pi * m_err * kT / scipy.constants.h**2, 3/2) * mpmath.polylog(5/2, z) * m_err * kT

    def norm(self, T, mu, Potential):
        volumeIntegral = integrate.nquad(lambda rx, ry, rz : self.momentumIntegral(np.array([rx, ry, rz]), T, mu, Potential), [[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf]])
        return volumeIntegral
    
    def averageMomentumSquared(self, T, mu, Potential):
        volumeIntegral = integrate.nquad(lambda rx, ry, rz : self.momentumSquaredIntegral(np.array([rx, ry, rz]), T, mu, Potential), [[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf]])
        norm = self.norm(T, mu, Potential)
        return volumeIntegral / norm