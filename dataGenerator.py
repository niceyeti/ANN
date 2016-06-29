"""
Script for generating 3d data for 2 classes, +1 and -1.

The +1 points will be centered about the origin, while the -1 points will be beyond the origin in a shell.
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random

fig = plt.figure(figsize=(14,10))
ax = fig.add_subplot(111, projection="3d")

X = [i for i in range(1,11)]
Y = [(random.randint(1,10)) for i in range(1,11)]
Z = [(2 * i) for i in range(1,11)]

#generate a bunch of points near the origin
X1 = [random.gauss(0,2) for i in range(0,30)]
Y1 = [random.gauss(0,2) for i in range(0,30)]
Z1 = [random.gauss(0,2) for i in range(0,30)]

#generate a bunch of points further from the origin in a shell
X2 = [random.gauss(0,2) * 4 for i in range(0,30)]
Y2 = [random.gauss(0,2) * 4 for i in range(0,30)]
Z2 = [random.gauss(0,2) * 4 for i in range(0,30)]

ax.scatter(X1,Y1,Z1, c="r", marker='o')
ax.scatter(X2,Y2,Z2, c="b", marker='o')

plt.show()