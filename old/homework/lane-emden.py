import numpy as np
import matplotlib.pyplot as plt

# Define polytropic indices
npoly = np.array([0.,1.,1.5,3.,5.]); nnpoly = len(npoly)

# Define xi-array
nxi = 1001; xi = np.linspace(0,20,nxi); dxi=xi[1]-xi[0]

# Rewrite equation in terms of y = \theta and z = dtheta/dxi
# Leads to two first order equations:
#   dy/dxi = z,  dz/dxi = - 2*(z/xi) - y^n. 
y = np.zeros(shape=(nxi,nnpoly))
z = np.zeros(shape=(nxi,nnpoly))

# Initial conditions: values near xi=0 from series expansion
y[0,:]=1.
for ipoly in range(0,nnpoly):
  y[1,:]= 1.-(xi[1]**2)/6. + npoly[ipoly]/120.*xi[1]**4
  z[1,:]= (2./3.) - y[1,:]**npoly[ipoly]

# Simple Eurler-integration
for ixi in range(2,nxi):
  for ipoly in range(0,nnpoly):
     z[ixi,ipoly] = z[ixi-1,ipoly] - (2./xi[ixi]*z[ixi-1,ipoly] + y[ixi-1,ipoly]**npoly[ipoly])*dxi
     y[ixi,ipoly] = y[ixi-1,ipoly] + z[ixi-1,ipoly]*dxi

# Plotting section
plt.plot(xi,y[:,0])
plt.plot(xi,y[:,1])
plt.plot(xi,y[:,2])
plt.plot(xi,y[:,3])
plt.plot(xi,y[:,4])
plt.plot(xi,xi*0,linestyle='dotted')

# Labels
plt.xlabel(r'$\xi$')
plt.ylabel(r'$\theta$')

# Beautyfying
plt.ylim(-.25,1)
plt.xlim(0,15)
