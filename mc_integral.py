#from __future__ import division, print_function
import random
import numpy as np
import matplotlib.pyplot as plt

random.seed(1234)
def f(x): 
    return (np.sin(x)**4*np.cos(x/3)**2+np.sin(x/4)**6)*np.exp(-x/8)*1.4

nMax = 10000
xmin = 0.
xmax = 10.
ymin = 0.
ymax = 1.
A = (xmax-xmin)*(ymax-ymin)

n = 0
N = 0
extractedX = []
extractedY = []
goodX = []
goodY = []
values = []

for i in range(0,nMax):
    N +=1
    thisX = random.uniform(xmin,xmax)
    thisY = random.uniform(ymin,ymax)
    extractedX.append(thisX)
    extractedY.append(thisY)
    if thisY < f(thisX):
        n +=1
        goodX.append(thisX)
        goodY.append(thisY)
    values.append(n/N*A)

print("n     =",n)
print("N     =",N)
print("n/N*A =",n/N*A)

w, h = plt.figaspect(1.)
plt.figure(figsize=(w,h), dpi=160)

plt.subplot(211)
x1 = np.arange(xmin,xmax,0.01)
plt.grid(True)
plt.xlabel('x',labelpad=0.5)
plt.ylabel('y',labelpad=0.5)
plt.scatter(extractedX, extractedY)
plt.scatter(goodX, goodY)
plt.plot(x1, f(x1), 'r--')

plt.subplot(212)
plt.grid(True)
plt.xlabel('n',labelpad=0.5)
plt.ylabel(u'n/N*A',labelpad=0.5)
plt.plot(range(0,nMax), values)

plt.show()