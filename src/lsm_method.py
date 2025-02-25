import numpy as np
from sampling_paths_generators import GBMPathGenerator
from polynomial_regression import polynomial_regression
from scipy import stats
import time
import csv
import os

def LSM(b, ObsGen, exercise_dates, payoff, an_rfr, degree): 
    assert isinstance(exercise_dates, list)
    sim_dates = exercise_dates if 0 in exercise_dates else [0] + exercise_dates
    ObsGen.generate_paths(sim_dates, b)
    mesh = ObsGen.paths()
    n_dates = len(exercise_dates)
    offset = 0 if 0 in exercise_dates else 1
    value = np.zeros(shape=(b, n_dates))
    # backward induction method starting from maturity
    for i_date in np.arange(n_dates - 1, -offset, -1):
        if i_date == n_dates - 1:
            # value at maturity is the payoff of the option if exercised
            value[..., i_date] = [payoff(mesh[:, j, i_date + offset]) for j in range(b)]
            continue
        discount = np.exp(-an_rfr * (exercise_dates[i_date + 1] - exercise_dates[i_date]))
        payoff_val = np.array([payoff(mesh[:, j, i_date + offset]) for j in range(b)])
        mask = payoff_val > 0 # distinguish in the money paths
        # first initialize values as discounted values from i_date + 1 for convenience
        value[:, i_date] = discount * value[:, i_date + 1]
        # regression on polynomial functions up to degree = degree
        cont_reg, poly_features = polynomial_regression(mesh[:, mask, i_date + offset].T, value[mask, i_date],
                                                 degree)
        
        value[mask, i_date] = np.maximum(payoff_val[mask], 
                                         cont_reg.predict(poly_features.transform(mesh[:, mask, i_date + offset].T)))
    # last step
    discount = np.exp(-an_rfr * exercise_dates[1 - offset])
    cont_value = discount * np.mean(value[:, 1 - offset])
    value = max(payoff(ObsGen.ci), cont_value) if 0 in exercise_dates else cont_value
    return value

def val_LSM(it,b, ObsGen, exercise_dates, payoff, an_rfr, degree):
    results = []
    for i in range(it):
        print('It {}-ésima'.format(i))
        results.append(LSM(b, ObsGen, exercise_dates, payoff, an_rfr, degree))
    return {'Price':np.mean(results), 'Std': np.std(results)}

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
    ci = [110] * 5
    f = lambda x: max(max(x)- k, 0)
    dates = [0, 1/3, 2/3, 1]
    ObsGen = GBMPathGenerator(ci, returns, cov_matrix)
    price = LSM(b=40000, ObsGen=ObsGen, exercise_dates=dates, payoff=f,an_rfr=rfr, degree=2)
    print(price)


