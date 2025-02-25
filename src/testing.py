from sampling_paths_generators import GBMPathGenerator
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
cov_matrix = np.eye(1)
ci = [1] 
returns = [0.5]
ObsGen = GBMPathGenerator(ci, returns, cov_matrix)
f = lambda x: ObsGen.cond_df(np.array(x), ObsGen.ci, 1)
plt.plot(np.linspace(-20,20,10000), [np.exp(x)*f(np.exp(x)) for x in np.linspace(-20,20,10000)], color='green')
plt.plot(np.linspace(-20,20,10000), stats.norm().pdf(np.linspace(-20,20,10000)), color='red')
plt.show()

cov_matrix = np.eye(7) * 0.4 ** 2
ci = [100] * 7
returns = [-0.02] * 7
ObsGen = GBMPathGenerator(ci, returns, cov_matrix)
ObsGen.generate_paths(dates=[0,1,2], N_paths=1000)
for i in range(7):
    for j in [1,2]:
        obs1 = ObsGen.GBMPaths[i,:,j]
        obs1 = np.log(obs1 / ObsGen.ci[i])
        obs1= (obs1 - (returns[i]-0.5 * ObsGen.cov_matrix[i,i])*j) / ObsGen.chol[i,i] / np.sqrt(j)
        stats.probplot(obs1, dist="norm", plot=plt)
        plt.show()
        pass