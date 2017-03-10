import numpy as np
import random

file_noise = open("oakland_part3_add_noise_1.node_features",'w')
with open("oakland_part3_mixed.node_features") as datafile:
    dataset = list(datafile)

    for i in range(0, len(dataset)):
        temp = dataset[i].strip().split()
        temp1 = [float(x) for x in temp]
        changeAtt = np.array(temp1[5:10])
        noise = 1 * np.random.rand(5)
        newAtt = changeAtt + noise
        newStr = []
        for j in newAtt:
            newStr.append(str(j))
        # print newAtt
        # print temp[0:5]
        newdata = temp[0:5] + newStr + temp[10:]
        new = ''
        for k in newdata:
            new = new + k + ' '
        new += '\n'
        file_noise.write(new)
file_noise.close()
