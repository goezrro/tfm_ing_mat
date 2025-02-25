import sys
import os
import time
# Obtener la ruta del directorio padre
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Agregar la carpeta padre a sys.path
sys.path.append(parent_dir)

import numpy as np
from scipy import stats
from sampling_paths_generators import GBMPathGenerator
from random_tree_mehotd import val_random_tree
from stochastic_mesh_method import Val_StochasMeshMethod
from lsm_method import val_LSM

def BS_vanilla_call(spot, strike, rfr, T, sigma):
    """Black-Scholes-Merton formula for european call options
    :param rfr: Annualised risk-free rate.
    :param T: time to maturity measured in years.
    :type T: float.    
    """
    d1 = (np.log(spot / strike)+(rfr + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return stats.norm.cdf(d1)*spot - stats.norm.cdf(d2) * strike * np.exp(-rfr * T)

if __name__ =='__main__':
    S0 = [50]    
    rfr = 0.06
    sigma = 0.2
    T=10
    k = 40
    f = lambda x: max(x[0]-k,0)
    print('Call europea: {}'.format(BS_vanilla_call(S0[0],k,rfr,T,sigma)))
    ObsGen = GBMPathGenerator(ci=np.array(S0), returns=np.array([rfr]), cov_matrix=np.array([sigma**2]))
    inicio = time.time()
    testDF_RTM = val_random_tree(it=20,b=8, dates=[0,2,4,6,8,10],ObsGen=ObsGen, payoff=f, rfr=rfr)
    fin = time.time()
    testMesh = Val_StochasMeshMethod(20,50,ObsGen=ObsGen,exercise_dates=[0,2,4,6,8,10],payoff=f,an_rfr=rfr,n_low_it=500)
    testLSM = val_LSM(50,1000,ObsGen,exercise_dates=[0,2,4,6,8,10],payoff=f,an_rfr=rfr,degree=3)
    pass
