import sys
import os
import itertools
import pandas as pd
import numpy as np
# Obtener la ruta del directorio padre
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Agregar la carpeta padre a sys.path
sys.path.append(parent_dir)


import time
import csv
from sampling_paths_generators import GBMPathGenerator
from lsm_method import val_LSM

if __name__ == '__main__':
    mat = [1,2]
    S0 = [36,38,40,44]    
    vols = [0.2, 0.4]
    rfr = 0.06
    # test1: put
    k = 40
    put = lambda x: max(k-x[0],0)
    df = None
    for comb in itertools.product(*[S0, mat, vols]):
        ObsGen = GBMPathGenerator([comb[0]], returns=[rfr], cov_matrix=np.array([comb[2] ** 2]))
        ini = time.time()
        value = val_LSM(25, 2 ** 15, ObsGen, exercise_dates=np.arange(0, comb[1] + 1/50, 1/50).tolist(),
                    payoff=put, an_rfr=rfr, degree=2)
        fin = time.time()
        value2 = val_LSM(25, 2 ** 15, ObsGen, exercise_dates=np.arange(0, comb[1] + 1/50, 1/50).tolist(),
                    payoff=put, an_rfr=rfr, degree=4)
        fin2 = time.time()
        results = {"S0": comb[0], "T": comb[1], "sigma": comb[2], "Price (Degree 2)": value['Price'], "Std. Error (Degree 2)": value['Std'], "Time (Degree 2) (s)": fin - ini,
                    "Price (Degree 4)": value2['Price'], "Std. Error (Degree 4)": value2['Std'], "Time (Degree 4) (s)": fin2 - fin}
        df = pd.DataFrame([results]) if df is None else pd.concat([df, pd.DataFrame([results])], ignore_index=True) 
    results_path = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(results_path, 'results_lsm')
    df.to_csv(os.path.join(results_path, 'put_50_remake.csv'))
        