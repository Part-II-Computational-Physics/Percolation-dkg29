#code for size distribution

import matplotlib.pyplot as plt
import numpy as np
rng = np.random.default_rng()
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib as mpl

#Define functions

#Determine the positions of the nearest neighbours of all sites
def boundaries(L):
    N=L**2
    nn=np.zeros((N,4),dtype=int)
    for i in range(L**2):
        nn[i, 0]=(i+1)%N
        nn[i, 1]=(i+N-1)%N
        nn[i, 2]=(i+L)%N
        nn[i, 3]=(i+N-L)%N
        if i%L == 0:
            nn[i, 1]=i+L-1
        if ((i+1)%L==0):
            nn[i, 0]=i-L+1
    return nn

#Generate the random order of site occupation
def permutation(L):
    order=np.arange(L**2)
    for i in range(L**2):
        j=rng.integers(i,L**2)
        temp=order[i]
        order[i]=order[j]
        order[j]=temp
    return order

#Return the position of a given sites' root and completes path compression
def findroot(ptr,i,vec):
    if ptr[i]<0: 
        #final output for root
        return i 
    else:
        #path compressive steps for the vectors and pointers
        vec[i] = disp(vec, i, ptr)
        ptr[i] = findroot(ptr,ptr[i],vec)
        #recursive step
        return findroot(ptr, ptr[i],vec) 
    
#Determine the displacement vector from a site to its root
def disp(vectors, i, ptr, total=[0,0]):
    if ptr[i] < 0:
        return total
    else:
        #recursive step
        return disp(vectors, ptr[i],ptr,total+vectors[i]) 
    
#Generates binomial distribution iteratively
def binomial(N,p):
    Binomial=np.zeros(N)
    nmax=round(p*N)
    Binomial[nmax-1]=1
    #formula for B(N,n,p) from B(N,n-1,p) 
    for i in range(nmax,N):
        Binomial[i]=(Binomial[i-1]*(N-i)*p)/((i+1)*(1-p))
    #formula for B(N,n,p) from B(N,n+1,p) 
    for i in range(0,nmax-1):
        j=nmax-i-2
        Binomial[j]=(Binomial[j+1]*(j+2)*(1-p))/((N-j-1)*p)
    C=np.sum(Binomial)+(Binomial[0]*(1-p)/(N*p))
    Binomial=Binomial/C
    #Excludes B(N,0,p) as this usually 0 or can be easily calculated
    return Binomial
        
#carries out a convolution to find Q(p)
def convolution(L, p_values, quantity):
    N=L**2
    Q_p=np.zeros([N])
    #p interval stored in p_values
    for n,p in enumerate(p_values):
        Binomial=binomial(N,p)
        #convolute and sum over binomial distribution for each p
        Q_p[n]=np.sum(Binomial*quantity)
    return Q_p #returns an array for each p in interval considered

def percolate(order,L):
    # main algorithm
    
    #initial setup
    N=L**2
    empty=-(N+1) 
    ptr=empty*np.ones(N, dtype=int) #pointer array
    vec=np.zeros((N,2), dtype=int) #vector array
    
    #Quantities to track
    Perc=False #True when percolation has occurred
    max_cluster_size=0 
    cluster_num=0 
    cluster_sizes_n=np.zeros(N)
    moments_of_inertia=np.zeros(N)
    max_moment_of_inertia=0
    centre_of_masses=np.zeros((N,2))
    
    #loop while adding sites
    for i in range(N):
        
        #occupy position s1 (with root r1)
        #forms cluster on its own
        r1=s1=order[i] 
        ptr[s1]=-1     
        
        #update quantities
        cluster_num+=1
        cluster_sizes_n[r1]=1
        
        #loop over nearest neighbours
        for j in range(4):
            
            s2=nn[s1, j] #jth nn of s1 (s2)
            
            #if a neighbouring site is occupied then amalgamate their clusters
            if ptr[s2] != empty:
                r2=findroot(ptr,s2,vec) #root of s2 (r2)
                #compare root sizes
                if (r2!=r1): 
                    cluster_num-=1 #increment quantity
                    
                    #weighted find - make the smaller of the two a subtree of the larger
                    if ptr[r1]>ptr[r2]: 
                        #cluster of r2 is larger than that of r1 so set vector of r1 to point to r2
                        
                        #first set root vector of r1 to point to r2 using vector addition
                        if j==0:
                            vec[r1,0]=(-disp(vec, s1, ptr, np.array([0,0]))[0]+1+disp(vec, s2, ptr, np.array([0,0]))[0])
                            vec[r1,1]=(-disp(vec, s1, ptr, np.array([0,0]))[1]+disp(vec, s2, ptr, np.array([0,0]))[1])

                        elif j==1:
                            vec[r1,0]=(-disp(vec, s1, ptr, np.array([0,0]))[0]-1+disp(vec, s2, ptr, np.array([0,0]))[0])
                            vec[r1,1]=(-disp(vec, s1, ptr, np.array([0,0]))[1]+disp(vec, s2, ptr, np.array([0,0]))[1])

                        elif j==2:
                            vec[r1,0]=(-disp(vec, s1, ptr, np.array([0,0]))[0]+disp(vec, s2, ptr, np.array([0,0]))[0])
                            vec[r1,1]=(-disp(vec, s1, ptr, np.array([0,0]))[1]+1+disp(vec, s2, ptr, np.array([0,0]))[1])

                        elif j==3:
                            vec[r1,0]=(-disp(vec, s1, ptr, np.array([0,0]))[0]+disp(vec, s2, ptr, np.array([0,0]))[0])
                            vec[r1,1]=(-disp(vec, s1, ptr, np.array([0,0]))[1]-1+disp(vec, s2, ptr, np.array([0,0]))[1])
                        
                        
                        #increment quantities
                        centre_of_masses[r1]=(-vec[r1]+centre_of_masses[r1]) #COM of the cluster of r1 measured from r2
                        new_COM=(cluster_sizes_n[r1]*centre_of_masses[r1]+cluster_sizes_n[r2]*centre_of_masses[r2])\
                                /(cluster_sizes_n[r1]+cluster_sizes_n[r2]) #calculating the new centre of mass
                        difference_r1=new_COM-centre_of_masses[r1]
                        difference_r2=new_COM-centre_of_masses[r2]
                        #calculating the new moment of inertia
                        moments_of_inertia[r2]=moments_of_inertia[r1]+cluster_sizes_n[r1]*np.dot(difference_r1,difference_r1)\
                                            +moments_of_inertia[r2]+cluster_sizes_n[r2]*np.dot(difference_r2,difference_r2)
                        moments_of_inertia[r1]=0
                        centre_of_masses[r2]=new_COM
                        centre_of_masses[r1]=[0,0]
                        
                        
                        
                        cluster_sizes_n[r1]=0
                        cluster_sizes_n[r2]=-(ptr[r2]+ptr[r1])
                        
                        #update pointers
                        ptr[r2]=ptr[r2]+ptr[r1] #add tree sizes
                        ptr[r1]=r2 #set pointer of r1 to r2
                        r1=r2 #set of root of r1 equal to r2

                    else:
                        #cluster of r1 is larger or equal than r1 so set r2 to point to r1
                        
                        #first set root vector of r2 to point to r1 using vector addition
                        if j==0:
                            vec[r2,0]=(-disp(vec, s2, ptr, np.array([0,0]))[0]-1+disp(vec, s1, ptr, np.array([0,0]))[0])
                            vec[r2,1]=(-disp(vec, s2, ptr, np.array([0,0]))[1]+disp(vec, s1, ptr, np.array([0,0]))[1])

                        elif j==1:
                            vec[r2,0]=(-disp(vec, s2, ptr, np.array([0,0]))[0]+1+disp(vec, s1, ptr, np.array([0,0]))[0])
                            vec[r2,1]=(-disp(vec, s2, ptr, np.array([0,0]))[1]+disp(vec, s1, ptr, np.array([0,0]))[1])

                        elif j==2:
                            vec[r2,0]=(-disp(vec, s2, ptr, np.array([0,0]))[0]+disp(vec, s1, ptr, np.array([0,0]))[0])
                            vec[r2,1]=(-disp(vec, s2, ptr, np.array([0,0]))[1]-1+disp(vec, s1, ptr, np.array([0,0]))[1])

                        elif j==3:
                            vec[r2,0]=(-disp(vec, s2, ptr, np.array([0,0]))[0]+disp(vec, s1, ptr, np.array([0,0]))[0])
                            vec[r2,1]=(-disp(vec, s2, ptr, np.array([0,0]))[1]+1+disp(vec, s1, ptr, np.array([0,0]))[1])
                            
                        #increment quantities
                        centre_of_masses[r2]=(-vec[r2]+centre_of_masses[r2]) #COM of the cluster of r2 measured from r1
                        new_COM=(cluster_sizes_n[r1]*centre_of_masses[r1]+cluster_sizes_n[r2]*centre_of_masses[r2])\
                                /(cluster_sizes_n[r1]+cluster_sizes_n[r2]) #new centre of mass
                        difference_r1=new_COM-centre_of_masses[r1]
                        difference_r2=new_COM-centre_of_masses[r2]
                        #new moment of inertia
                        moments_of_inertia[r1]=moments_of_inertia[r1]+cluster_sizes_n[r1]*np.dot(difference_r1,difference_r1)\
                                            +moments_of_inertia[r2]+cluster_sizes_n[r2]*np.dot(difference_r2,difference_r2)
                        moments_of_inertia[r2]=0
                        centre_of_masses[r1]=new_COM
                        centre_of_masses[r2]=[0,0]
                        
                        
                        cluster_sizes_n[r2]=0
                        cluster_sizes_n[r1]=-(ptr[r1]+ptr[r2])
                        
                        #update pointer array
                        ptr[r1]=ptr[r1]+ptr[r2] #add tree sizes
                        ptr[r2]=r1 #set pointer of r2 to r1
                    
                    
                    #if the cluster is now the largest update quantity
                    if -ptr[r1]>max_cluster_size:
                        max_cluster_size=-ptr[r1] #increment quantity
                    else:
                        pass
                    
                                       
                    #if thecluster has the largest moment of inertia
                    if moments_of_inertia[r1]>max_moment_of_inertia:
                        max_moment_of_inertia=moments_of_inertia[r1]
                    else:
                        pass
                
                #if the new site connects two sites belonging to the same cluster check for percolation
                else:
                    v1=disp(vec,s1,ptr)
                    v2=disp(vec,s2,ptr)
                    displacement=abs(v2-v1)
                    #if a component is larger than 1 percolate
                    if displacement[0] > 1 or displacement[1] > 1:
                        Perc=True
                        percolation_values.append((i+1)/N)
                        
            
            else:
                pass
                
        
        #update quantities after each site addition
        percolation_probability.append((i+1.)/N) 
        largest_cluster_size_n.append(max_cluster_size)
        cluster_num_n.append(cluster_num)
        cluster_sizes[i]=cluster_sizes_n
        
        #calculate chi_n and P_n depending on whether the lattice has percolated
        if Perc == True:
            chi_n.append((np.sum(cluster_sizes_n**2)-max_cluster_size**2)/(i+1))
            P_n.append(max_cluster_size/(i+1))
            if np.sum(cluster_sizes_n)-max_cluster_size==0: #the system only contains the percolated cluster 
                correlation_length_n.append(0)
            else:
                correlation_length_n.append(np.sqrt((np.sum(cluster_sizes_n*moments_of_inertia)\
                                        -max_cluster_size*max_moment_of_inertia)\
                                        /(np.sum(cluster_sizes_n**2)-max_cluster_size**2)))
        else:
            chi_n.append(np.sum(cluster_sizes_n**2)/(i+1))
            P_n.append(0)
            correlation_length_n.append(np.sqrt(np.sum(cluster_sizes_n*moments_of_inertia)/np.sum(cluster_sizes_n**2)))

#quantities to track
P_n=[] #Order parameter
chi_n=[] #susceptibility
percolation_probability=[] #num of steps/N
largest_cluster_size_n=[] #largest cluster at each step
percolation_values=[] #steps at which percolation has occured
cluster_num_n=[] #number of clusters at each step
correlation_length_n=[]

cluster_sizes=np.zeros((100,100)) #sizes of clusters
nn=boundaries(10)
order=permutation(10)
percolate(order,10)

sizes=[]
frequencies=[]
#determine the sizes of clusters present at each step and their frequency
for i in cluster_sizes:
    unique, counts = np.unique(i, return_counts=True) 
    sizes.append(unique[1:])
    frequencies.append(counts[1:])
   
    
final_frequencies=np.zeros((100,101))
#generate an array containing the frequency for a given step and cluster size to plot
for i,item in enumerate(sizes):
      for j,value in enumerate(item):
            final_frequencies[int(i),int(value)]=frequencies[int(i)][int(j)]


#defining an appropritae color map
rainbow = mpl.colormaps['hsv']
newcolors = rainbow(np.linspace(0, 50*1/int(np.amax(final_frequencies)), int(np.amax(final_frequencies))))
white = np.array([256/256, 256/256, 256/256, 1])
black = np.array([0,0,0, 1])
green = np.array([0,1,0, 1])
blue = np.array([0,0,1, 1])
newcolors[0, :] = white
newcolors[1, :] = black
newcolors[2, :] = green
newcolors[3, :] = blue
newcmap = ListedColormap(newcolors)


#plotting
cluster_size, step_number = np.meshgrid(np.linspace(1,5, 101), np.linspace(1, 5, 100))
fig, ax = plt.subplots()
c = ax.pcolormesh(step_number, cluster_size, final_frequencies, cmap=newcmap)
plt.xlabel('Step number n/20')
plt.ylabel('Cluster size/20')
plt.axvline(0.5927*5,color='black',linestyle='dotted',label='$p_c$')
fig.colorbar(c, ax=ax, label='number of clusters')
plt.legend(loc='lower right')
plt.show()
