import numpy as np
from scipy import stats
import copy as cp
from sampling_paths_generators import BMPathGenerator, GBMPathGenerator

""" This code is made to built a tree with nodes which are related by a given  conditional 
distribution. Initially make for geometric brownian motion observations,  but could be generalized.
"""
class Nodo:
    def __init__(self, valor, identificador=str(1)):
        self.valor = valor
        self.hijos = []
        self.id = identificador # añadido para el método RT con DF

class MVtree:
    def __init__(self, b, dates, ObsGen):
        self.raiz = None
        self.b = b # número de hijos por nodo (branching parameter)
        assert dates[0] == 0, 'La primera fecha de la lista debe ser 0 (presente)'
        self.dates = dates # determina los niveles que se van a generar
        self.n = len(self.dates) - 1 
        self.ObsGen = ObsGen # objeto para generar los hijos
        self.raiz = Nodo(ObsGen.ci)
        self.generar_arbol(self.raiz)
    def insertar(self, valor, nivel=0):
        if self.raiz is None:
            self.raiz = Nodo(valor)
        else:
            self._insertar_recursivo(self.raiz, valor, nivel)

    def _insertar_recursivo(self, nodo, valor, nivel):
        if nivel >= self.n - 1:
            print(f"No se puede insertar, se alcanzó la profundidad máxima: {self.n}")
            return False

        if len(nodo.hijos) < self.b:
            nodo.hijos.append(Nodo(valor))
            return True
        else:
            # Si el nodo ya tiene el máximo de hijos, intentamos insertar en los hijos
            for hijo in nodo.hijos:
                if self._insertar_recursivo(hijo, valor, nivel + 1):
                    return True
            return False  # Si todos los hijos están llenos, no se puede insertar

    def recorrer(self):
        resultado = []
        self._recorrer_recursivo(self.raiz, resultado)
        return resultado

    def _recorrer_recursivo(self, nodo, resultado):
        if nodo is not None:
            resultado.append(nodo.valor)
            for hijo in nodo.hijos:
                self._recorrer_recursivo(hijo, resultado)
    
    def generar_arbol(self, padre, i=0):
        """Genera el árbol recursivamente usando el generador de observaciones."""
        if i >= self.n:
            return
        else:
            hijos = self.ObsGen.generate_paths(dates=[self.dates[i], self.dates[i + 1]], N_paths=self.b, ci=padre.valor, save=False)[:, :, -1]
            for j, hijo_valor in enumerate(hijos.T):
                hijo = Nodo(hijo_valor)
                padre.hijos.append(hijo)
                self.generar_arbol(hijo, i+1)


        

if __name__=='__main__':
    # testing the implementation
    ObsGen = GBMPathGenerator(ci=np.array([10]), returns=np.array([0.04]), cov_matrix=np.array([0.2]))
    MarkovTree = MVtree(b=2, dates=[0,1,2], ObsGen=ObsGen)
    MarkovTree = MVtree(b=1000, dates=[0,1,2], ObsGen=ObsGen)

    test_sample_ci = MarkovTree.raiz.hijos[434].valor
    test_sample = np.array([MarkovTree.raiz.hijos[434].hijos[j].valor for j in range(1000)])
    print(stats.kstest((np.log(test_sample / test_sample_ci) - np.exp(0.04 * 1))/1/0.2, cdf=stats.norm.cdf).pvalue)
    pass



    