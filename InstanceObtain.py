# Define a class "instance" which includes position, timeStep, label and features
import numpy as np

class Instance:
    """
        Read from the data file, each istance has 4 attributes.
        pos: position of the cloud point clouds
        timeStep: time step of the point
        label: actual label
        features: 10 features with a bias feature
    """
    # def __init__(self, pos = [0, 0, 0], timeStep = 0, label = 0, features = np.zeros(10)):
    #     self.pos = pos
    #     self.timeStep = timeStep
    #     self.label = label
    #     self.features = features
    def __init__(self, dataline):
        self.pos = np.array(dataline[0:3])
        self.timeStep = int(dataline[3])
        self.label = int(dataline[4])
        self.features = np.array(dataline[5:])
