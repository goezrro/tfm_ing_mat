import numpy as np
from markov_chain_tree import MVtree, Nodo
from sampling_paths_generators import GBMPathGenerator
from scipy import stats
import time

class BaseRandomTreeMethod(MVtree):
    def __init__(self, b, dates, ObsGen, payoff, rfr):
        super().__init__(b, dates, ObsGen)
        # se asume que bajo la medida risk neutral el return de todos los activos será igual e igual a la tasa libre de riesgo 
        self.rfr = rfr
        assert callable(payoff)
        self.compute_high_estimator(self.raiz, payoff, 0)
        self.compute_low_estimator(self.raiz, payoff, 0)

    def compute_high_estimator(self, nodo, payoff, d):
        """Compute the high estimator on a recursive way""" 
        if not nodo.hijos:
            nodo.high_estimator = payoff(nodo.valor)
            return
        dt = self.dates[d+1] - self.dates[d]
        discount = np.exp(- self.rfr * dt)
        for j in range(self.b):
            self.compute_high_estimator(nodo.hijos[j], payoff, d + 1)
        # el factor de descuento está fuera porque se va a considerar algo constante y conocido (simplificación)
        if d ==0:
            pass
        high_estimators = [hijo.high_estimator for hijo in nodo.hijos]
        nodo.high_estimator = max(payoff(nodo.valor), discount * np.mean(high_estimators))
    # usamos la misma construcción que en el papel de Broadie y Glasserman
    def compute_low_estimator(self, nodo, payoff, d):
        if not nodo.hijos:
            nodo.low_estimator = payoff(nodo.valor)
            return    
        dt = self.dates[d+1] - self.dates[d]
        discount = np.exp(- self.rfr * dt)
        for j in range(self.b):
            self.compute_low_estimator(nodo.hijos[j], payoff, d+1)
        payoff_val = payoff(nodo.valor)
        low_estimators = [discount * hijo.low_estimator for hijo in nodo.hijos]
        low_cont_values = np.array([np.mean([low_estimators[j] for j in range(self.b) if j != k]) for k in range(self.b)])
        nodo.low_estimator = np.mean(np.where(payoff_val > low_cont_values, payoff_val, low_estimators))

class DepthFirsRTM(MVtree):
    def __init__(self, b, dates, ObsGen, payoff, rfr):
        assert callable(payoff)
        self.payoff = payoff
        # se asume que bajo la medida risk neutral el return de todos los activos será igual e igual a la tasa libre de riesgo 
        self.rfr = rfr   
        super().__init__(b, dates, ObsGen)             
    
    def generar_arbol(self, padre, i=0):
        """"
        En este caso, con la generación del árbol se van calculando de manerar dinámica los 
        estimadores, y se van descartando los sucesores 
        """
        if i == len(self.dates) - 1:
            #print('procesando nodo {}, de valor {}:'.format(padre.id, padre.valor))   
            padre.low_estimator = padre.high_estimator = self.payoff(padre.valor)
            return
        for j in range(self.b):
            # generamos un único hijo hasta llegar a la profundidad deseada
            self.ObsGen.generate_paths(dates=np.array([self.dates[i], self.dates[i + 1]]),
                                        N_paths=1, ci=padre.valor)
            hijo = Nodo(self.ObsGen.GBMPaths[:, 0, -1], identificador=padre.id+str(j+1))
            padre.hijos.append(hijo)
            self.generar_arbol(hijo, i+1)
        dt = self.dates[i+1] - self.dates[i]
        discount = np.exp(-self.rfr * dt)
        #print('procesando nodo {}, de valor {}'.format(padre.id, padre.valor))
        payoff_val = self.payoff(padre.valor)
        high_estimators = [hijo.high_estimator for hijo in padre.hijos]
        padre.high_estimator =max(payoff_val, discount * np.mean(high_estimators))
        low_estimators = np.array([discount * hijo.low_estimator for hijo in padre.hijos])
        low_cont_values = np.array([np.mean([low_estimators[j] for j in range(self.b) if j != k]) for k in range(self.b)])
        padre.low_estimator = np.mean(np.where(payoff_val > low_cont_values, payoff_val, low_estimators)) 
        padre.hijos.clear() # borro los hijos para liberar espacio       

def val_random_tree(it, b, dates, ObsGen, payoff, rfr):
    high=[]
    low=[]
    for i in range(it):
        print('It {}-ésima'.format(i))
        aux = BaseRandomTreeMethod(b, dates, ObsGen, payoff, rfr)
        high.append(aux.raiz.high_estimator)
        low.append(aux.raiz.low_estimator)
    return {'HighEst': np.mean(high), 'HighStd':np.std(high),
            'LowEst': np.mean(low), 'LowStd':np.std(low)}


def single_stock_call(spot, strike):
    return max(spot[0] - strike, 0)

def payoff_max_two_stocks(spots, strike):
    return max(max(spots) - strike, 0)

def BS_vanilla_call(spot, strike, rfr, T, sigma):
    """Black-Scholes-Merton formula for european call options
    :param rfr: Annualised risk-free rate.
    :param T: time to maturity measured in years.
    :type T: float.    
    """
    d1 = (np.ln(spot / strike)+(rfr + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return stats.norm.cdf(d1)*spot - stats.norm.cdf(d2) * strike * np.exp(-rfr * T)

if __name__=='__main__':
    #ObsGen = GBMPathGenerator(ci=np.array([10]), returns=np.array([0.04]), cov_matrix=np.array([0.2]))
    # opción call bermuda evaluable cada año de aquí a 3 años con strike 10
    #f = lambda x: single_stock_call(spot=x, strike=10)
    #testBaseRTM = BaseRandomTreeMethod(b=50, dates=[0,1,2,3],ObsGen=ObsGen, payoff=f)
    #print('The low biased estimator is: {}'.format(testBaseRTM.raiz.low_estimator))
    #print('The high biased estimator is: {}'.format(testBaseRTM.raiz.high_estimator))
    
    f = lambda x: payoff_max_two_stocks(spots=x, strike=100)
    # varios activos
    ro12 = 0.3
    sigma = 0.2
    cov_matrix = np.array([[sigma ** 2, ro12 * sigma ** 2], [ro12 * sigma ** 2, sigma ** 2]])
    ObsGen = GBMPathGenerator(ci=np.array([110, 110]), returns=np.array([-0.05, -0.05]), cov_matrix=cov_matrix)
    inicio = time.time()
    testBaseRTM_multiple_assets = BaseRandomTreeMethod(b=100, dates=[0,1,2,3],ObsGen=ObsGen, payoff=f, rfr=0.05)
    fin = time.time()
    print('The low biased estimator is: {}'.format(testBaseRTM_multiple_assets.raiz.low_estimator))
    print('The high biased estimator is: {}'.format(testBaseRTM_multiple_assets.raiz.high_estimator))
    print('Tiempo: {}s'.format(fin - inicio))
    #input()
    ObsGen = GBMPathGenerator(ci=np.array([110, 110]), returns=np.array([-0.05, -0.05]), cov_matrix=cov_matrix)
    inicio = time.time()
    testDF_RTM = DepthFirsRTM(b=100, dates=[0,1,2,3],ObsGen=ObsGen, payoff=f, rfr=0.05)
    fin = time.time()
    print('The low biased estimator is: {}'.format(testDF_RTM.raiz.low_estimator))
    print('The high biased estimator is: {}'.format(testDF_RTM.raiz.high_estimator))
    print('Tiempo: {}s'.format(fin - inicio))
