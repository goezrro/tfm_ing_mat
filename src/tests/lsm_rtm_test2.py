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
from random_tree_mehotd import BaseRandomTreeMethod, DepthFirsRTM
from lsm_method import LSM


if __name__ == '__main__':
    sigma = 0.2
    ro = 0.3
    # test 2: call on max of 5 assets
    cov_matrix = np.array([[sigma ** 2, ro * sigma ** 2, ro * sigma ** 2,ro * sigma ** 2,ro * sigma ** 2],
                           [ro * sigma ** 2, sigma ** 2,ro * sigma ** 2,ro * sigma ** 2,ro * sigma ** 2],
                           [ro * sigma ** 2, ro * sigma ** 2,sigma ** 2,ro * sigma ** 2,ro * sigma ** 2],
                           [ro * sigma ** 2, ro * sigma ** 2,ro * sigma ** 2,sigma ** 2,ro * sigma ** 2],
                           [ro * sigma ** 2, ro * sigma ** 2,ro * sigma ** 2,ro * sigma ** 2,sigma ** 2]])
    returns = [-0.05] * 5
    
    rfr = 0.05
    k = 100
    f = lambda x: max(max(x)- k, 0)
    dates = [0, 1/3, 2/3, 1]
    list_val = [80,90,100,110,120]
    for val_ini in list_val:
        ci = [val_ini] * 5     
        results = list()   
        for i in range(50):
            ObsGen = GBMPathGenerator(ci, returns, cov_matrix)
            inicio = time.time()
            test1 = LSM(2 ** 15, ObsGen, exercise_dates=dates,
                    payoff=f, an_rfr=rfr, degree=2)
            fin = time.time()
            test2 = LSM(2 ** 15, ObsGen, exercise_dates=dates,
                    payoff=f, an_rfr=rfr, degree=3)
            fin2 = time.time()
            results.append({'S0':val_ini, "price_degree2": test1, 'time_2': fin - inicio, "price_degree3": test2,  'time_3': fin2 - fin})
        results_path = os.path.dirname(os.path.abspath(__file__))

        results_path = os.path.join(results_path, 'results_lsm_replica')
        with open(os.path.join(results_path, "rtm_test2_{}.csv".format(val_ini)), mode="w", newline="") as file:
            fieldnames = ["S0", "price_degree2", "time_2", 'price_degree3', 'time_3']
            writer = csv.DictWriter(file, fieldnames=fieldnames)        
            writer.writeheader()  # Escribir encabezados
            writer.writerows(results)  # Escribir filas