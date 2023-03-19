#calculate volume of n dimensional ball using monte carlo integration
#calculates volumes for all dimensions from n=2 to n=15

import numpy as np
import math as m
import matplotlib.pyplot as plt

v=[]
#creates arrays v and n for the analytical plot of the volume function
for n in range(2,16):
    v.append((np.pi**(n/2))/(m.gamma((n/2)+1)))
n=np.arange(2,16,1)

#number of points to randomly generate in the upper quadrant=10,000
n_pts=10**5

#this uniformly randomly generates a number between 0 and 1 for all possible dimensions
rannum=np.random.random((n_pts,15))

vols=[]
dims=[]
#these loops use the randomly generated points to calculate the volume of each unit n-ball
for dim in range(1,15):
    #count=number of points under the curve
    count=0
    for pts in range(0,n_pts):
        i=0
        tot=0
        #add the squares of each random number  and check if the sum is under 1
        while i <= dim:
            tot+=(rannum[pts][i])**2
            i+=1
        if tot < 1.0:
            count+=1
    #the volume is calculated as the fraction of the points under the curve multiplied by 2^(dimension)
    vol=(count/n_pts)*2**(dim+1)
    vols.append(vol)
    dims.append(dim+1)
    print("volume of",(dim+1),"dimensional ball is: ",vol)

#plot the numerical solution and the analytical solutions together to compare
plt.plot(dims,vols, color='teal', label="Numerical")
plt.plot(n,v, color='goldenrod', label="Analytical")
plt.xlabel("Dimensions of ball, n")
plt.ylabel("Volume of unit n-ball")
plt.title("Volume of unit n-ball over dimension, n")
plt.legend(loc="upper right")
plt.xticks(n)
plt.show()