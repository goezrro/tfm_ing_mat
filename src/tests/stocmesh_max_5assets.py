import sys
import os
# Obtener la ruta del directorio padre
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Agregar la carpeta padre a sys.path
sys.path.append(parent_dir)

import numpy as np
import time
import csv
from sampling_paths_generators import GBMPathGenerator
from stochastic_mesh_method import StochasticMeshMethodInterp, StochasticMeshMethod

def call_geom_avg(spots, strike):
    return max(np.prod(np.array(spots)) ** (1/ len(spots)) - strike, 0)

if __name__ == '__main__':
    cov_matrix = np.eye(5) * 0.2 ** 2
    ci = [90] * 5
    returns = [-0.05] * 5
    rfr = 0.05
    k = 100
    f = lambda x: max(max(x)- k, 0)
    n_paths_list = [500, 1000, 2000]
    it = 25
    for n_paths in n_paths_list:
        results = list()
        results_int = list()
        n_low = 10 * n_paths
        for i in range(it):
            print('Iteración {}-ésima, n_paths={}'.format(i+1, n_paths))
            ObsGen = GBMPathGenerator(ci, returns, cov_matrix)
            # ini = time.time()
            # high_estimator, low_estimator = StochasticMeshMethod(b=n_paths, ObsGen=ObsGen, exercise_dates=[0,1,2,3], payoff=f, an_rfr=rfr, n_low_it=n_low)
            int = time.time()
            high_estimator_int, low_estimator_int = StochasticMeshMethodInterp(b=n_paths, ObsGen=ObsGen, exercise_dates=[0,1,2,3], payoff=f, an_rfr=rfr, n_low_it=n_low)
            fin = time.time()
            # results.append({'It':i, "high_estimator": high_estimator, 'low_estimator':low_estimator, 'time': int - ini})
            results_int.append({'It':i, "high_estimator": high_estimator_int, 'low_estimator':low_estimator_int, 'time': fin - int})
        
        results_path = os.path.dirname(os.path.abspath(__file__))

        results_path = os.path.join(os.path.join(results_path, 'results_mesh_method'), 'max5assets_spot90')
        # with open(os.path.join(results_path, "datos_{}_{}.csv".format(n_paths,n_low)), mode="w", newline="") as file:
        #     fieldnames = ["It", "high_estimator", "low_estimator", "time"]
        #     writer = csv.DictWriter(file, fieldnames=fieldnames)        
        #     writer.writeheader()  # Escribir encabezados
        #     writer.writerows(results)  # Escribir filas
        with open(os.path.join(results_path, "datos_{}_{}_int.csv".format(n_paths,n_low)), mode="w", newline="") as file:
           fieldnames = ["It", "high_estimator", "low_estimator", "time"]
           writer = csv.DictWriter(file, fieldnames=fieldnames)        
           writer.writeheader()  # Escribir encabezados
           writer.writerows(results_int)  # Escribir filas
