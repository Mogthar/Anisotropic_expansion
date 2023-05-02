#%% imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants
import scipy.integrate as integrate
import potential
import time
import mpmath

m_err = 167.259 * scipy.constants.atomic_mass
#%% functions

def MaxwellBoltzmann(p, r, T, mu, Potential):
    fugacity = np.exp(mu / (scipy.constants.Boltzmann * T))
    return fugacity * np.exp(-Potential.value(r) / (scipy.constants.Boltzmann * T) - np.sum(p**2) / (2 * m_err * scipy.constants.Boltzmann * T))

def BoseEinstein(p, r, T, mu, Potential):
    fugacity = np.exp(mu / (scipy.constants.Boltzmann * T))
    return 1 / (np.exp(Potential.value(r) / (scipy.constants.Boltzmann * T) + np.sum(p**2) / (2 * m_err * scipy.constants.Boltzmann * T) / fugacity) - 1)

def DistributionNorm(Distribution, Potential, T, mu, tolerance):
    p_limit = np.sqrt(2 * m_err * scipy.constants.Boltzmann * T * tolerance)
    r_limit = Potential.integrationLimit(T, tolerance)

    return integrate.nquad(lambda px, py, pz, rx, ry, rz : Distribution(np.array([px, py, pz]), np.array(rx, ry, rz), T, mu, Potential),
                           [[-p_limit, p_limit], [-p_limit, p_limit], [-p_limit, p_limit], [-r_limit[0], r_limit[0]], [-r_limit[1], r_limit[1]], [-r_limit[2], r_limit[2]]])

def AverageMomentumSquared(Distribution, Potential, T, mu, tolerance = 10):
    p_limit = np.sqrt(2 * m_err * scipy.constants.Boltzmann * T * tolerance)
    r_limit = Potential.integrationLimit(T, tolerance)

    integral_x = integrate.nquad(lambda px, py, pz, rx, ry, rz : Distribution(np.array([px, py, pz]), np.array([rx, ry, rz]), T, mu, Potential) * px**2,
                                [[-p_limit, p_limit], [-p_limit, p_limit], [-p_limit, p_limit], [-r_limit[0], r_limit[0]], [-r_limit[1], r_limit[1]], [-r_limit[2], r_limit[2]]])
    integral_y = integrate.nquad(lambda px, py, pz, rx, ry, rz : Distribution(np.array([px, py, pz]), np.array([rx, ry, rz]), T, mu, Potential) * py**2,
                                [[-p_limit, p_limit], [-p_limit, p_limit], [-p_limit, p_limit], [-r_limit[0], r_limit[0]], [-r_limit[1], r_limit[1]], [-r_limit[2], r_limit[2]]])
    integral_z = integrate.nquad(lambda px, py, pz, rx, ry, rz : Distribution(np.array([px, py, pz]), np.array([rx, ry, rz]), T, mu, Potential) * pz**2,
                                [[-p_limit, p_limit], [-p_limit, p_limit], [-p_limit, p_limit], [-r_limit[0], r_limit[0]], [-r_limit[1], r_limit[1]], [-r_limit[2], r_limit[2]]])

    norm = DistributionNorm(Distribution, Potential, T, mu, tolerance)
    return np.array([integral_x, integral_y, integral_z]) / norm

#%% test usual convergence conditions for the two distributions
temperature = 200e-9
mu = -10 * scipy.constants.Boltzmann * temperature
omega = 2 * np.pi * np.array([290, 14, 230])
trapping_potential = potential.HarmonicPotential(omega)

t1 = time.time()
print(AverageMomentumSquared(MaxwellBoltzmann, trapping_potential, temperature, mu, 10))
t2 = time.time()
print(t2 - t1)
print(AverageMomentumSquared(BoseEinstein, trapping_potential, temperature, mu, 10))
t3 = time.time()
print(t3 - t2)

# %%
