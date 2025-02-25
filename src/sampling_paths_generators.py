# standard libraries import
import numpy as np
import numpy.random as rng
import matplotlib.pyplot as plt
from scipy import stats

# random number generator, fixed seed for reproducibility
#rng.seed(666)

class BMPathGenerator:
    """Standard brownian motion generator, for a given number random sources, set of dates and 
    covariance matrix"""
    def __init__(self, cov_matrix=None) -> None:
        """
        :param cov_matrix: 
        :type cov_matrix: numpy.array
        """
        self.cov_matrix = cov_matrix
        self.N_sources = cov_matrix.shape[0] if cov_matrix is not None else 1

        # TODO: handle the inputs
            # asegurar que la matriz de covarianzas es semidefinida positiva

    def generate_paths(self, dates, N_paths):
        """
        :param dates:
        :type dates: list
        :param N_paths:
        :type N_paths: int
        """
        # cholesky factorization of the covariance matrix
        B = np.linalg.cholesky(self.cov_matrix) if self.N_sources > 1 else np.sqrt(self.cov_matrix)
        self.chol = B
        # Compute the random increments
        dt = np.sqrt(np.diff(dates))        
        Z = rng.randn(self.N_sources, N_paths, len(dt))
        paths = np.zeros(shape=(self.N_sources, N_paths, len(dates)))
        # correlate the independent observations
        aux_Z = np.zeros_like(Z)
        for j in range(N_paths):
            aux_Z[:, j, :] = np.dot(B, Z[:, j, :])
        self.corr_inc = aux_Z
        # time variance factor
        A = np.zeros(shape=(dt.shape[0], dt.shape[0]))
        for i in range(dt.shape[0]):
            A[i, :] = [dt[i] * (j>=i) for j in range(len(dt))]  
        paths[:, :, 1:] = [np.dot(aux_Z[j,...], A) for j in range(self.N_sources)]

        #W = np.zeros(shape=(N_paths, len(dt) + 1))
        #W[:, 1:] = np.dot(Z, A)
        self.BMpaths = np.array(paths)

class GBMPathGenerator(BMPathGenerator):
    """Geometric Brownian motion generator following notation from Glasserman. This is, 
    the stochastic solution proccess for dS = mu*S*dt+sigma*S*dW(t)"""
    def __init__(self, ci, returns, cov_matrix=None):
        """
        :param ci: vector de valores iniciales de los activos
        :type ci: list
        :param returns: drift vector for the N assets
        :type returns: list
        """
        super().__init__(cov_matrix)
        # handle inputs
        self.ci = np.array(ci)
        self.returns = np.array(returns)
    
    def generate_paths(self, dates, N_paths, ci=None, save=True):
        "se supone que la primera fecha es a la que corresponden los ci"
        dates=np.array(dates)
        ci=self.ci if ci is None else ci
        super().generate_paths(dates, N_paths)
        square_vols = np.array([self.cov_matrix[i,i] for i in range(self.cov_matrix.shape[0])])  if self.N_sources > 1 else self.cov_matrix      
        
        aux = np.zeros_like(self.BMpaths)
        aux[:, :, 1:] = ci[:, None, None] * np.exp((self.returns - 0.5 * square_vols)[:, None, None] * 
                                                             (dates[1:]-dates[0])[None, None, :] + self.BMpaths[..., 1:])
        aux[..., 0] = ci[:, None] # save initial value to the beginnig of every path for each asset
        if save:
            self.GBMPaths = aux
            return
        else:
            return aux
    
    def cond_df(self, Si, Sj, dt):
        """
        Densidades condicionadas ---- SOLO TIENE SENTIDO SI SON INDEPENDIENTES LOS ACTIVOS, SINO LAS CUENTAS NO SON VÁLIDAS
        :param Si: valor final
        :param Sj: valor inicial
        :param dt: intervalo de tiempo
        """
        mu = (self.returns - 0.5 * np.diag(self.cov_matrix)) * dt + np.log(Sj)
        sigma = dt * self.cov_matrix
        return 1 / np.prod(Si) * stats.multivariate_normal.pdf(np.log(Si), mean=mu, cov=sigma)
    
    def inped_df(self, Si, Sj, dt):
        return np.prod([1 / (self.chol[i,i] * np.sqrt(dt) * Si[i]) * 
                        stats.norm.pdf((np.log(Si[i]/Sj[i]) - (self.returns[i] - 0.5 * self.cov_matrix[i,i]) * dt) / (np.sqrt(dt) * self.chol[i,i])) for i in range(len(Si))])
    
    def paths(self):
        return self.GBMPaths

if __name__=='__main__':
    # testing BM implementation
    # cholesky factor
    chol = np.array([[0.5, 0], [0.4, 0.8]])
    cov_matrix = np.dot(chol,chol.transpose())
    testBM = BMPathGenerator(cov_matrix=np.array(cov_matrix))
    testBM.generate_paths(dates=np.linspace(0,1,50), N_paths=10000)
    # some sanity checks
    alpha = 0.05 # confidence threshold to accept normality of the observations
    for i in range(testBM.N_sources):
        assert np.all([stats.kstest(testBM.BMpaths[i, :, j + 1]/np.linspace(0,1,50)[j + 1]/cov_matrix[i,i], cdf=stats.norm.cdf).pvalue < alpha for j in range(49)])
    # testing GBM implementation
    returns = [0.12, 0.07]
    ci = [2,8]
    testGBM = GBMPathGenerator(ci=ci, returns=returns, cov_matrix=np.array(cov_matrix))
    testGBM.generate_paths(dates=np.linspace(0,1,50), N_paths=10000)
    # sanity chcks
    for i in range(testBM.N_sources):
        assert np.all([stats.kstest((np.log(testGBM.GBMPaths[i, :, j + 1] / ci[i]) - np.exp(returns[i]*np.linspace(0,1,50)[j + 1]))/np.linspace(0,1,50)[j + 1]/cov_matrix[i,i], cdf=stats.norm.cdf).pvalue < alpha for j in range(49)])
    print('Todo como se esperaba jefe')