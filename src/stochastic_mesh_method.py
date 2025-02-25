import numpy as np
from markov_chain_tree import MVtree, Nodo
from sampling_paths_generators import GBMPathGenerator
from random_tree_mehotd import BaseRandomTreeMethod
from scipy import stats
from scipy.interpolate import LinearNDInterpolator, RBFInterpolator
import time
import pandas as pd
import csv
import os

def cont_value(Si, date_i, S_next, V_next, g_next, date_next, ObsGen):
    dt = date_next - date_i
    b = S_next.shape[1]
    cond_probs = np.array([ObsGen.cond_df(S_next[:, j], Si, dt) for j in range(b)])
    return 1 / b * np.dot(cond_probs, (V_next / g_next).T)

def StochasticMeshMethod(b, ObsGen, exercise_dates, payoff, an_rfr, n_low_it=50):       
    ObsGen.generate_paths(exercise_dates, b)
    mesh = ObsGen.paths()
    n_dates = len(exercise_dates)
    g_eval = np.zeros(shape=(b, n_dates))
    high_estimator = np.zeros(shape=(b, n_dates))
    offset = 0 if 0 in exercise_dates else 1
    for i_date in np.arange(n_dates - 1, -offset, -1):        
        if i_date == n_dates - 1:
            high_estimator[:, i_date] = [payoff(mesh[:, j, i_date]) for j in range(b)]
            continue
        dt = exercise_dates[i_date + 1] - exercise_dates[i_date]
        discount = np.exp(-an_rfr * dt)
        cond_vals = np.array([[ObsGen.cond_df(mesh[:, k, i_date + 1], mesh[:, i, i_date], dt) for k in range(b)] for i in range(b)])
        cond_sums = cond_vals.sum(axis=0, keepdims=True)  # Suma por columna (definición de ref 66 - Broadie and Glasserman)
        g_eval[:, i_date + 1] = 1/ b * cond_vals.sum(axis=0)
        payoff_eval = np.array([payoff(mesh[:, j, i_date]) for j in range(b)])
        continuation_values = np.dot(cond_vals/ cond_sums, high_estimator[:, i_date + 1]) * discount
        high_estimator[:, i_date] = np.where(payoff_eval > 0, np.maximum(payoff_eval, continuation_values), high_estimator[:, i_date + 1] * discount)
    discount = np.exp(-an_rfr * (exercise_dates[1] - exercise_dates[0]))
    high_est = max(payoff(ObsGen.ci), np.mean(high_estimator[:, 1]) * discount) if 0 in exercise_dates else np.mean(high_estimator[:,0])
    def low_estim_comp(idx, path):
        discount_factor = np.exp(-an_rfr * exercise_dates[idx])
        low = payoff(path[..., idx + offset])
        if idx == n_dates - 1 or low >= np.exp(-an_rfr * (exercise_dates[idx + 1] - exercise_dates[idx])) * cont_value(path[:, idx], exercise_dates[idx], mesh[:, :, idx + 1], high_estimator[:, idx + 1], g_eval[:, idx + 1], exercise_dates[idx + 1], ObsGen):
            # última fecha nos quedamos con el payoff ya
            return low * discount_factor
        else:
            return low_estim_comp(idx + 1, path)
    path = ObsGen.generate_paths(exercise_dates, n_low_it, save=False)
    low_estim = [low_estim_comp(0, path[:, j, :]) * np.exp(-an_rfr * exercise_dates[0]) for j in range(n_low_it)]
    low_estimator = np.mean(low_estim)
    # último paso distinto 
    return high_est, low_estimator

def Val_StochasMeshMethod(it,b, ObsGen, exercise_dates, payoff, an_rfr, n_low_it):
    high=[]
    low=[]
    for i in range(it):
        print('It {}-ésima'.format(i))
        res_high, res_low = StochasticMeshMethod(b, ObsGen, exercise_dates, payoff, an_rfr, n_low_it)
        high.append(res_high)
        low.append(res_low)
    return {'HighEst': np.mean(high), 'HighStd':np.std(high),
            'LowEst': np.mean(low), 'LowStd':np.std(low)}


def StochasticMeshMethodInterp(b, ObsGen, exercise_dates, payoff, an_rfr, n_low_it=50):       
    ObsGen.generate_paths(exercise_dates, b)
    mesh = ObsGen.paths()
    n_dates = len(exercise_dates)
    weights = np.zeros((b, b, n_dates - 1))
    high_estimator = np.zeros((b, n_dates))

    offset = 0 if 0 in exercise_dates else 1

    for i_date in np.arange(n_dates - 1, -offset, -1):
        discount = np.exp(-an_rfr * exercise_dates[i_date])
        if i_date == n_dates - 1:
            high_estimator[:, i_date] = [payoff(mesh[:, j, i_date]) * discount for j in range(b)]
            continue
        cond_vals = np.array([[ObsGen.cond_df(mesh[:, k, i_date + 1], mesh[:, j, i_date], exercise_dates[i_date + 1] - exercise_dates[i_date]) for k in range(b)] for j in range(b)])
        cond_sums = cond_vals.sum(axis=0, keepdims=True)  # Suma por fila
        weights[:, :, i_date] = (cond_vals) / cond_sums  # Normalización
        # Cálculo de los valores esperados
        continuation_values = np.dot(weights[:, :, i_date], high_estimator[:, i_date + 1])
        payoff_val = np.array([payoff(mesh[:, j, i_date]) * discount for j in range(b)])
        mask = payoff_val > 0
        high_estimator[:, i_date] = np.where(mask, np.maximum(payoff_val, continuation_values), continuation_values)
    # último paso distinto 
    high_est = max(payoff(ObsGen.ci), np.mean(high_estimator[:, 1])) if 0 in exercise_dates else np.mean(high_estimator[:,0])
    interp = [LinearNDInterpolator(mesh[..., j].T, high_estimator[..., j]) for j in range(1, n_dates - 1)]
    def low_estim_comp(path):
        low = np.exp(-an_rfr * exercise_dates[0]) * payoff(path[..., offset])
        if exercise_dates[0] == 0 and low >= high_est:
                return low
        for idx in range(1, n_dates):
            if idx == n_dates -1:
                return low
            elif low >= interp[idx - 1](path[..., idx]):
                return low
            low = np.exp(-an_rfr * exercise_dates[idx + 1]) * payoff(path[..., idx + offset + 1])
        
        return low
    path = ObsGen.generate_paths(exercise_dates, n_low_it, save=False)
    low_estim = [low_estim_comp(path[:, j, :]) for j in range(n_low_it)]
    low_estimator = np.mean(low_estim)
    return high_est, low_estimator




def call_geom_avg(spots, strike):
    return max(np.prod(np.array(spots)) ** (1/ len(spots)) - strike, 0)
if __name__ == '__main__':
    cov_matrix = np.eye(5) * 0.2 ** 2
    ci = [90] * 5
    returns = [-0.05] * 5
    rfr = 0.05
    k = 100
    f = lambda x: max(max(x)- k, 0)
    ObsGen = GBMPathGenerator(ci, returns, cov_matrix)
    high_estimator, low_estimator = StochasticMeshMethod(50, ObsGen, exercise_dates=[0,1,2,3], payoff=f, an_rfr=rfr, n_low_it=500)
    print(high_estimator)
    print(low_estimator)
    #benchmark = BaseRandomTreeMethod(4, np.arange(0,1.1,0.1),ObsGen=ObsGen, payoff=f, rfr=rfr)
    # n_paths_list = [50, 100, 150, 200, 250, 300]
    n_paths_list = [50]
    it = 10
    for n_paths in n_paths_list:
        results = list()
        results_int = list()
        n_low = 10 * n_paths
        for i in range(it):
            print('Iteración {}-ésima, n_paths={}'.format(i+1, n_paths))
            ObsGen = GBMPathGenerator(ci, returns, cov_matrix)
            ini = time.time()
            high_estimator, low_estimator = StochasticMeshMethod(b=n_paths, ObsGen=ObsGen, exercise_dates=np.arange(0,1.1,0.1), payoff=f, an_rfr=rfr, n_low_it=n_low)
            int = time.time()
            high_estimator_int, low_estimator_int = StochasticMeshMethodInterp(b=n_paths, ObsGen=ObsGen, exercise_dates=np.arange(0,1.1,0.1), payoff=f, an_rfr=rfr, n_low_it=n_low)
            fin = time.time()
            results.append({'It':i, "high_estimator": high_estimator, 'low_estimator':low_estimator, 'time': int - ini})
            results_int.append({'It':i, "high_estimator": high_estimator_int, 'low_estimator':low_estimator_int, 'time': fin - int})
        
        results_path = os.getcwd()
        results_path = os.path.join(results_path, 'results_mesh_method')
        with open(os.path.join(results_path, "datos_{}_{}.csv".format(n_paths,n_low)), mode="w", newline="") as file:
            fieldnames = ["It", "high_estimator", "low_estimator", "time"]
            writer = csv.DictWriter(file, fieldnames=fieldnames)        
            writer.writeheader()  # Escribir encabezados
            writer.writerows(results)  # Escribir filas
        with open(os.path.join(results_path, "datos_{}_{}_int.csv".format(n_paths,n_low)), mode="w", newline="") as file:
            fieldnames = ["It", "high_estimator", "low_estimator", "time"]
            writer = csv.DictWriter(file, fieldnames=fieldnames)        
            writer.writeheader()  # Escribir encabezados
            writer.writerows(results_int)  # Escribir filas
