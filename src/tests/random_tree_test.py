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


if __name__ == '__main__':
    sigma = 0.2
    ro = 0.3
    # test 1: call on max of 2 assets
    # cov_matrix = np.array([[sigma ** 2, ro * sigma ** 2],[ro * sigma ** 2, sigma ** 2]])
    # returns = [-0.05] * 2
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
    b=50
    list_val = [80,90,100,110,120]
    for val_ini in list_val:
        ci = [val_ini] * 5     
        results = list()   
        for i in range(50):
            ObsGen = GBMPathGenerator(ci, returns, cov_matrix)
            inicio = time.time()
            baseRTM = BaseRandomTreeMethod(b=b, dates=dates,ObsGen=ObsGen, payoff=f, rfr=rfr)
            fin = time.time()
            results.append({'S0':val_ini, "high_estimator": baseRTM. raiz.high_estimator, 'low_estimator':baseRTM.raiz.low_estimator, 'time': fin - inicio})
        results_path = os.path.dirname(os.path.abspath(__file__))

        results_path = os.path.join(results_path, 'results_base_rtm')
        with open(os.path.join(results_path, "results_base_RTM_fiveassets_{}.csv".format(val_ini)), mode="w", newline="") as file:
            fieldnames = ["S0", "high_estimator", "low_estimator", "time"]
            writer = csv.DictWriter(file, fieldnames=fieldnames)        
            writer.writeheader()  # Escribir encabezados
            writer.writerows(results)  # Escribir filas
    
    #input()
    # inicio = time.time()
    # testDF_RTM = DepthFirsRTM(b=b, dates=dates,ObsGen=ObsGen, payoff=f, rfr=rfr)
    # fin = time.time()
    # print('The low biased estimator is: {}'.format(testDF_RTM.raiz.low_estimator))
    # print('The high biased estimator is: {}'.format(testDF_RTM.raiz.high_estimator))
    # print('Tiempo: {}s'.format(fin - inicio))