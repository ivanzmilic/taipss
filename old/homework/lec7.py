import numpy as np
import matplotlib.pyplot as plt
import math

# Load the data file
stars = np.loadtxt("yaletrigplx.dat.txt",unpack=True)

# Description of the data:
# column 1: star ID number
# column 2: apparent V magnitude
# column 3: observed B-V color
# column 4: observed parallax (in arcsec)
# column 5: uncertainty in parallax (in milliarcsec)

# Compute distance from parallax
parallax = stars[3]; d = 1./parallax

# Absolute magnitude
M = stars[1] + 5.0 - 5.0 * np.log10(d)

# Luminosity
Lsun = 3.83e26; Msun = 4.83
L = Lsun / 10**(0.4*(M - Msun))

# Compute temperature using the color
BV = stars[2]
T = 4600*(1./(0.92*BV + 1.7) + 1./(0.92*BV+0.62))

# Compute radius
sigma = 5.67e-8
R = (L/(4.*math.pi*sigma*T**4))**.5

# Recast in km instead of m
R = R/1e3

T1 = np.linspace(2.5e3,4e3,100)
logL1 = np.linspace(22,26,100)
logT1 = logL1/5.6

# Plot
plt.scatter(T,L,s=2)

# Beautify
plt.xscale('log'); plt.yscale('log')

# Invert x-axis and set limits
#plt.gca().invert_xaxis() 
#plt.xlim([2e4,2.5e3]) 
plt.xlim([2.5e3,2e4]) 
#plt.ylim([2.5e3,2e4]) 

# Labels
#plt.xlabel('T [K]')
plt.xlabel('L [W]')
#plt.ylabel('R [km]')
#plt.ylabel('L [W]')
plt.ylabel('T [K]')
