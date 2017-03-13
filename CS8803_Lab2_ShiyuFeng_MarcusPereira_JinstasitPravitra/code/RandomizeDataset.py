import numpy as np
import random
import csv
from InstanceObtain import Instance

# Read all data

data = []
with open("oakland_part3_add_noise_1.node_features") as datafile:
    dataset = list(datafile)
    random.shuffle(dataset)

n = len(dataset)
n_train = int(round(n * 0.7))
n_test = n - n_train
print n, n_train, n_test

file_train = open("training_set_noise_1.node_features",'w')
for i in range(0, n_train):
    file_train.writelines(dataset[i])
file_train.close()
file_test = open("test_set_noise_1.node_features", 'w')
for j in range(n_train, n):
    file_test.write(dataset[j])
file_test.close()
    # for row in dataset:
    #     temp = row[0].strip().split()
    #     temp = [float(x) for x in temp]
    #     print temp
        # instance = Instance(temp)
        # data.append(instance)

# random.shuffle(data)
# print data[0].pos
# # Randomly generate training set and test set
# n = 126219
# n_train = int(round(n * 0.7))
# n_test = n - n_train
#
# training_set = data[0: n_train]
# test_set = data[n_train:]
