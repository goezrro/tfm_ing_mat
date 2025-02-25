from sampling_paths_generators import GBMPathGenerator
from random_tree_mehotd import BaseRandomTreeMethod, DepthFirsRTM
from stochastic_mesh_method import StochasticMeshMethod, StochasticMeshMethodInterp
import numpy as np

""" Here we are testing all the implementations using a Bermudan Call on a single underlying.
Option expires in 3 years and can be exercised at any of 10 equally spaced exercise opportunities: m = 10
and ti = 0.3 i for i = 1,..,m. Payoff max(S(ti)-K,0). S described as GBM(r-d, sigma ** 2), con S(0)=100, 
sigma=0.2, r=5% and d=10%"""
if __name__=='__main__':
    cov_matrix = np.array([0.2 ** 2])
    ci = np.array([100]) 
    returns = np.array([-0.05]) 
    rfr = 0.05
    k = 100
    f = lambda x: max(x[0]-k,0)
    exercise_dates = np.array([0.3 * i for i in range(11)])
    ObsGen = GBMPathGenerator(ci,returns,cov_matrix)
    print('1')
    high_estim_sto, low_estim_sto= StochasticMeshMethod(b=100, ObsGen=ObsGen, 
                                                       exercise_dates=exercise_dates, payoff=f, 
                                                       an_rfr=rfr, n_low_it=1000)
    print('2')
    high_estim_stoint, low_estim_stoint= StochasticMeshMethodInterp(b=100, ObsGen=ObsGen, 
                                                       exercise_dates=exercise_dates, payoff=f, 
                                                       an_rfr=rfr, n_low_it=1000)
    print('3')
    BinLat = BaseRandomTreeMethod(b=3, dates=exercise_dates, ObsGen=ObsGen, payoff=f, rfr=rfr)
    print('4')
    DepthBinLat = DepthFirsRTM(b=3, dates=exercise_dates, ObsGen=ObsGen, payoff=f, rfr=rfr)
    pass
    