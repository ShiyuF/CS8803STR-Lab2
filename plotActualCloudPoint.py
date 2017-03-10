# Plot actual cloud points
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from InstanceObtain import Instance

# Generate a Plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlim(70, 280)
ax.set_ylim(70, 280)
ax.set_zlim(-10, 35)
ax.set_title("Actual labels for point clouds")
# Read files line by line
plotVeg = np.empty((1,3))
# plotVeg = [[]]
# print(plotVeg)
plotWire = np.zeros(3)
plotPole = np.zeros(3)
plotGround = np.zeros(3)
plotFacade = np.zeros(3)

with open("oakland_part3_mixed.node_features") as data:
    while True:
        line = data.readline().strip().split()
        line = [float(x) for x in line]
        if len(line) == 0:
            break

        # Split a line into pos, timeStep, label and features
        instance = Instance(line)
        pos = instance.pos
        pos = pos[None, :]
        # print(pos.shape)
        label = instance.label

        # Assign a color to each class
        if label == 1004:   # Veg
            c = 'g'
            # plotVeg = np.vstack((plotVeg, pos))
            plotVeg = np.concatenate((plotVeg, pos))
            # print(plotVeg.shape)
        elif label == 1100:
            c = 'k'
            plotWire = np.vstack((plotWire, pos))
        elif label == 1103:
            c = 'r'
            plotPole = np.vstack((plotPole, pos))
        elif label == 1200:
            c = 'b'
            plotGround = np.vstack((plotGround, pos))
        elif label == 1400:
            c = 'y'
            plotFacade = np.vstack((plotFacade, pos))

        # print(timeStep)
        # Plot the scatter points
        # ax.scatter(pos[0], pos[1], pos[2], s=1, c=c, depthshade=False, linewidth=0)
# Plot scatter points
classes = ['plotVeg', 'plotWire', 'plotPole', 'plotGround', 'plotFacade']
color = ['g', 'k', 'r', 'b', 'y']
i = 0
for x in classes:
    temp = eval(x)
    # temp = temp[1:]
    ax.scatter(temp[1:, 0], temp[1:, 1], temp[1:, 2], s=5, c=color[i], depthshade=True, linewidth=0)
    i += 1

plt.show()
