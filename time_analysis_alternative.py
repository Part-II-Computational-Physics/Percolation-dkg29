#time analysis for Newman-Ziff with colour plotting

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

#data for run times at different L
times=np.array([[0.021144300000014483, 0.020367950000036217, 0.020130349999999454, 0.020371909999994386, 0.020473340000035024, 0.01916467000000921, 0.018173190000015892, 0.017826670000022204, 0.018577820000064093, 0.01782395999994151],
      [0.118489250000016, 0.08877284000000145, 0.07505968999998913, 0.07319015000000491, 0.07397215999999389, 0.07394446000002972, 0.07470816999998534, 0.08829921000001377, 0.07470378000004985, 0.07281676000000062],
      [0.3765489299999899, 0.39440064000000347, 0.400384229999986, 0.36497532999999294, 0.45968467000000146, 0.3607336699999905, 0.35897869999998877, 0.3775528499999837, 0.37894325000002027, 0.36600711000005504],
      [1.234272030000011, 1.211006929999985, 1.7029907999999978, 1.408701189999988, 1.345242200000007, 1.361156609999989, 1.4892179599999964, 1.2782775099999981, 1.2186601799999834, 1.1326958599999444],
      [7.564240340000003, 7.075176259999966, 5.628156970000032, 6.608010319999972, 7.525167869999995, 6.896420019999982, 6.4805001700000044, 6.539834190000056, 5.90611644999999, 5.887235880000003]])

#values of L considered
L=np.array([7,10,15,20,30,])
N=L**2

mean_times=[]
error=[]
#averaging
for i in times:
    mean_times.append(np.mean(i))
    error.append(np.std(i)/(10)**0.5)
    
#curve fitting
def objective(x, A, n, b):
    return A*x**n+b

popt, pcov=curve_fit(objective,N,mean_times)
A,n,b=popt
p_sigma = np.sqrt(np.diag(pcov))

error=np.array(error)

#plotting
plt.text(70, 8, '$t = A*N^n+b$')
plt.text(70, 5.8, '$n = {n:.3f}\pm{error:.3f}$'.format(n=n,error=p_sigma[1]))
plt.text(70, 4.2, '$A = (9.6\pm0.2)*10^{-5}$')
plt.text(70, 3, '$b = (-0.01\pm0.02)$')
plt.xlabel('Lattice size N')
plt.ylabel('Runtime t [s]')
plt.xscale('log')
plt.yscale('log')
plt.plot(N,A*N**n+b,linestyle='dotted',color='k',label='Fit')
plt.errorbar(N,mean_times,yerr=error*10,marker='.',linestyle='none',label='data (Errorbarsx10)',color='orange')
plt.legend(loc='lower right')
plt.show()
