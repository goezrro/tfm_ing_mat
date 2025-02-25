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
import math
def geom_avg(x):
    aux = 1
    for num in x:
        aux *= num ** (1/len(x))
    return aux 
if __name__ == '__main__':
    sigma = 0.2
    ro = 0.3
    # test 2: call on geom avg of 5 assets
    corr_mat = np.zeros((5,5))
    corr_mat[0,4]=corr_mat[0,1]=0.8
    corr_mat[0,2]=corr_mat[1,2]=0.2
    corr_mat[0,3]=corr_mat[1,3]=0.3
    corr_mat[1,4]=corr_mat[2,3]=0.1
    corr_mat[2,4]=corr_mat[3,4]=0.7
    corr_mat = (corr_mat + corr_mat.T) /2 +np.eye(5)
    vol=[0.4,0.4,0.4,0.3,0.3]
    vol = np.diag(vol)
    cov_matrix = np.dot(np.dot(vol, corr_mat),vol)
    rfr = 0.05
    div_yields = np.array([0.06,0.06,0.08,0.08,0.08])
    returns = rfr - div_yields   
    k = 100
    f = lambda x: max(geom_avg(x) - k, 0)
    dates = [0, 1/3, 2/3, 1]
    b=50
    list_val = [70,80,90,100,110,120,130]
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
        with open(os.path.join(results_path, "results_base_RTM_geomavg_fiveassets_{}.csv".format(val_ini)), mode="w", newline="") as file:
            fieldnames = ["S0", "high_estimator", "low_estimator", "time"]
            writer = csv.DictWriter(file, fieldnames=fieldnames)        
            writer.writeheader()  # Escribir encabezados
            writer.writerows(results)  # Escribir filas