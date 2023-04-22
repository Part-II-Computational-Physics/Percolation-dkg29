#plots for the determination of beta and gamma

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

#import data for chi and Pinfinity

datac30 = np.loadtxt('datac30.csv', delimiter=',')
datac50 = np.loadtxt('datac50.csv', delimiter=',')
datac60 = np.loadtxt('datac60.csv', delimiter=',')
datac90 = np.loadtxt('datac90.csv', delimiter=',')
datac100 = np.loadtxt('datac100copy.csv', delimiter=',')
  
def Q(N,p):
    #generate binomial coeffients using recursive algorithm  
    Binomial=np.zeros(N)
    nmax=round(p*N)
    Binomial[nmax-1]=1
    for i in range(nmax,N):
        Binomial[i]=(Binomial[i-1]*(N-i)*p)/((i+1)*(1-p))
    for i in range(0,nmax-1):
        j=nmax-i-2
        Binomial[j]=(Binomial[j+1]*(j+2)*(1-p))/((N-j-1)*p)
    C=np.sum(Binomial)+(Binomial[0]*(1-p)/(N*p))
    Binomial=Binomial/C
    return Binomial
        

def mean_var(L, p, data):
    #perform convolution and determine the mean and error for 10 data points
    N=L**2
    Q_fact=Q(N,p)
    out=np.zeros(10)
    final=np.zeros(2)
    for i,j in enumerate(data):
        value=np.sum(Q_fact*j)
        out[i]=value
    final[0]=np.mean(out) 
    final[1]=np.std(out)/(10**0.5)
    
    return final

def plottingP(p,L,values,beta):
    #returns points to plot for (L^(beta/nu))*Pinfinity
    points=np.zeros((2,len(p)))
    for i,j in enumerate(p):
        points[0,i]=((L**(beta/nu))*mean_var(L,j,values)[0])
        points[1,i]=((L**(beta/nu))*mean_var(L,j,values)[1])
    return points
    
def plottingChi(p,L,values,gamma):
    #returns points to plot for (L^(-gamma/nu))*chi
    points=np.zeros((2,len(p)))
    for i,j in enumerate(p):
        points[0,i]=((L**(-gamma/nu))*mean_var(L,j,values)[0])
        points[1,i]=((L**(-gamma/nu))*mean_var(L,j,values)[1])
    return points

def scalingFactor(L,nu,pc,p):
    #returns L*|p-pc|/pc
    values=np.zeros(len(p))
    for i,j in enumerate(p):
        if j>=pc:
            values[i]=(L**(1/nu))*((j-pc)/pc)
        else:
            values[i]= (L**(1/nu))*((-j+pc)/pc)
    return values

#parameter values
pc=0.5927
nu=1.329
beta=0.123
gamma=2.35

#values of p to plot
p=np.linspace(0.5,0.63,100)


#plotting for chi
plt.errorbar(scalingFactor(30,nu,pc,p),plottingChi(p,30,datac30,gamma)[0],yerr=plottingChi(p,30,datac30,gamma)[1],marker='.',linestyle='none',label='L=30',color='green')
plt.errorbar(scalingFactor(60,nu,pc,p),plottingChi(p,60,datac60,gamma)[0],yerr=plottingChi(p,60,datac60,gamma)[1],marker='.',linestyle='none',label='L=60',color='red')
plt.errorbar(scalingFactor(90,nu,pc,p),plottingChi(p,90,datac90,gamma)[0],yerr=plottingChi(p,90,datac90,gamma)[1],marker='.',linestyle='none',label='L=90',color='orange')
plt.errorbar(scalingFactor(100,nu,pc,p),plottingChi(p,100,datac100,gamma)[0],yerr=plottingChi(p,100,datac100,gamma)[1],marker='.',linestyle='none',label='L=100',color='blue')
plt.legend(loc='lower right')
plt.ylabel('$χ(p)*L^{-γ/ν}$')
plt.xlabel('$L^{1/ν}*|p-p_c|/p_c$')
plt.text(4, 0.2, '$γ = 2.35\pm0.05$')
plt.show()
