import sys
import os
# Obtener la ruta del directorio padre
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Agregar la carpeta padre a sys.path
sys.path.append(parent_dir)

import numpy as np
from sampling_paths_generators import GBMPathGenerator

def call_geom_avg(spots, strike):
    return max(np.prod(np.array(spots)) ** (1/ len(spots)) - strike, 0)
def test1():
    """ El objetivo del test es contrastar el valor que obtenemos para una opción call europea a vencimiento 1 año sobre el 
    promedio geométrico de 7 activos no correlados (p_ij = 0), cada uno modelado por un GBM(r- d, sigma ** 2)
    donde r = 3% (risk free rate), d = 5% (dividend yield) y sigma=0.4 (volatilidad). El precio que se da en el Glasserman
    es 2.42. El precio inicial es 100 para todos los activos y el strike es 100.
    """ 
    cov_matrix = np.eye(7) * 0.4 ** 2
    ci = [100] * 7
    returns = [-0.02] * 7
    rfr = 0.03
    T = 1
    discount = np.exp(-rfr * T)
    k = 100
    f = lambda x: call_geom_avg(x, k)
    N_paths= 2 ** 16
    prices = list()
    for i in range(50):    
        ObsGen = GBMPathGenerator(ci, returns, cov_matrix) 
        ObsGen.generate_paths(dates=[0,T], N_paths=N_paths)
        prices.append(np.mean([f(ObsGen.paths()[:, j, -1]) for j in range(N_paths)]) * discount)
    final_price = np.mean(prices)
    print(final_price)
    print(abs(final_price-2.42) / 2.42 * 100)
    print(np.std(prices))
def test2():
    """ El objetivo del test es contrastar el valor que obtenemos para una opción call europea a vencimiento 3 años sobre el 
    máximo de 5 activos no correlados (p_ij = 0), cada uno modelado por un GBM(r- d, sigma ** 2)
    donde r = 5% (risk free rate), d = 10% (dividend yield) y sigma=0.2 (volatilidad). El precio que se da en el Glasserman
    es 23.052. El precio inicial es 100 para todos los activos y el strike es 100.
    """ 
    benchmark = [14.586, 23.052, 32.685]
    cis = [90,100,110]
    id = 2
    cov_matrix = np.eye(5) * 0.2 ** 2
    ci = [cis[id]] * 5
    returns = [-0.05] * 5
    rfr = 0.05
    T = 3
    discount = np.exp(-rfr * T)
    k = 100
    f = lambda x: max(max(x)- k, 0)
    N_paths= 2 ** 16
    prices = list()
    for i in range(50):    
        ObsGen = GBMPathGenerator(ci, returns, cov_matrix) 
        ObsGen.generate_paths(dates=[0,T], N_paths=N_paths)
        prices.append(np.mean([f(ObsGen.paths()[:, j, -1]) for j in range(N_paths)]) * discount)
    final_price = np.mean(prices)
    print(final_price)
    print(abs(final_price-benchmark[id]) / benchmark[id] * 100)
    print(np.std(prices))


if __name__ == '__main__':
    test1()
    test2()