import math
import numpy as np
from matplotlib import cm
from matplotlib import animation
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3

# Twice as wide as it is tall.
fig = plt.figure(figsize=plt.figaspect(0.5))

# Parameters
a = 1		    # Data generated as y = a*x + b + r
b = 1		    # where r is a normally distributed random
sigma = 1	    # number with mean 0 and std sigma
xLim = 5	    # +- limits for generated x-values
eta = 0.02	    # Learning rate
epochs = 200    # Learning epochs
interval = 20   # Time interval between frames

# Number of data points
n = 100

# Generating data points
x = np.linspace(-xLim, xLim,n)			# Linearly spaced x-values	
X = np.vstack( (np.ones(n), x) )		# X matrix, [1; x1]
coeff = np.array( [b, a] ).reshape(2,1)		# Array with the parameters a and b
r = np.random.normal(0, sigma*sigma, n)		# Random noice
y = np.dot( coeff.transpose(), X ) + r		# Observed outputs for each x value
# Plotting data points
ax1 = fig.add_subplot(1,2,1)
data = ax1.plot(x,y.flatten(),'bo')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_xlim(-5, 5)
ax1.set_ylim(math.floor(y.min()), math.ceil(y.max()))

# Resolution for error surface
W0_lim = (-1, 3)
W1_lim = (-1, 3)
dW = 0.05

# Creating matrices
W0 = np.arange(W0_lim[0],W0_lim[1]+dW,dW)	# Array of w0 values
W1 = np.arange(W1_lim[0],W1_lim[1]+dW,dW)	# Array of w1 values
W0, W1 = np.meshgrid(W0,W1)			# W0 nd W1 matrices
Eav = np.zeros(W0.size).reshape(W0.shape)	# Emptty matrix for Eav
iMax, jMax = W0.shape				# Dimensions for Eav

# Average error energy
for i in range(iMax):
	for j in range(jMax):
		W = np.array( [ W0[i,j], W1[i,j] ] ).reshape(2,1)	# Weight combination
		yHat = np.dot( W.transpose(), X )	# Calculating model outputs
		e = y - yHat				# Error signal
		Eav[i,j] = np.mean(e**2)		# Average error energy

# Plotting error surface
ax2 = fig.add_subplot(1,2,2, projection='3d')
surf = ax2.plot_surface(W0, W1, Eav, rstride=2, cstride=4, cmap=cm.coolwarm, linewidth=0, 
	antialiased=True)
surf.set_alpha(0.75)
ax2.set_xlabel(r'$w_0$')
ax2.set_ylabel(r'$w_1$')
ax2.set_zlabel(r'$E_{av}$')
ax2.set_xlim3d(W0_lim)
ax2.set_ylim3d(W1_lim)

# Initializing the animation
def init():
	model.set_data([],[])		# Model
	path.set_data([],[])		# Path on the error surface
	path.set_3d_properties([])	# Path on the error surface
	return model,path,

#Gradient descent
def gradientDescent(self):
	global W, w0, w1, Eav_plot
	# Current model and error
	yHat = np.dot( W.transpose(), X ) 	# Model output on training data
	e = y - yHat				# error signal
	Eav = np.mean(e**2)			# Current average error
	# 3D coordinates on the error surface
	w0 = np.append(w0, W[0])		# Appending w0
	w1 = np.append(w1, W[1])		# Appending w1
	Eav_plot = np.append(Eav_plot, Eav)	# Appending the error for current model
	# Updating plots
	model.set_data(x, yHat)			# Current model
	path.set_data(w0, w1)			# Path on the error surface
	path.set_3d_properties(Eav_plot)	# Path on the error surface
	# Adjusting weights
	dW = - eta / n * np.dot( X, e.transpose() )
	W -= dW
	return model,path,

W = np.random.normal(1, 2, 2).reshape(2,1)
w0 = np.array([])
w1 = np.array([])
Eav_plot = np.array([])
model, = ax1.plot([],[],'r-')
path, = ax2.plot([],[],[],'ko-')
	
anim = animation.FuncAnimation(fig, gradientDescent, frames=epochs, init_func=init, 
	interval=interval, blit=True, repeat=False)

plt.show()

