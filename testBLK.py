import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time
import random

from BLK import BayesianLinearRegression
from InstanceObtain import Instance

def main():
    # initialize mean and covariance of prior
    iterations = 1
    validateStep = 10000
    a = 5
    mu_p = np.zeros(10)
    sigma_p = a * np.eye(10)
    sigma = 1

    # one vs. all strategy: create 5 classifiers
    # classes = [1004, 1100, 1103, 1200, 1400]
    # blk = []
    # for target in classes:
    #     blk.append(BayesianLinearRegression(target, mu_p, sigma_p, sigma))
    #     print("Generated a BLK model for class %i." % (target,))
    target1 = 1004
    target2 = 1400
    blk = BayesianLinearRegression(target1, target2, mu_p, sigma_p, sigma)
    print("Generated a BLK model.")

    # read each line and train through BLK
    validateError = []
    itertime = np.zeros(iterations)
    training_time = 0
    for k in range(0, iterations):
        with open("training_set.node_features") as data:
        # with open("training_set_noise_0.5.node_features") as data:
            data = list(data)
            random.shuffle(data)
            for t in range(0, len(data)):
                # read file and format instance
                line = data[t].strip().split()
                line = [float(x) for x in line]

                instance = Instance(line)

                # set blk model and update
                # for i in range(0, 5):
                #     blk[i].updateRule(instance)
                start_train = time.time()
                blk.updateRule(instance)
                stop_train = time.time()
                trainTime = stop_train - start_train
                training_time += trainTime
                # print("Updated\n")

                # Each validateStep test model on validation set.
                if t % validateStep == 0:
                    blk.set_final_parameters()
                    validateNumInstance = 0
                    validateCorrect = 0
                    error = 0

                    with open("test_set.node_features") as validate:
                    # with open("test_set_noise_0.5.node_features") as validate:
                        while True:
                            validateline = validate.readline().strip().split()
                            validateline = [float(x) for x in validateline]
                            if len(validateline) == 0:
                                break
                            validate_instance = Instance(validateline)
                            if validate_instance.label == target1 or validate_instance.label == target2:
                                validateNumInstance += 1

                            validate_mean, validate_covariance = blk.prediction(validate_instance)
                            if validate_instance.label == target1:
                                error += (1 - validate_mean) ** 2 / 2
                            elif validate_instance.label == target2:
                                error += (-1 - validate_mean) ** 2 / 2

                    validateError.append(error[0])
                    print("Finished one validatation. The squared error is %f." % error)

    # plot error convergence curve
    fig2 = plt.figure()
    ax2 = fig2.gca()
    ax2.set_title("Error rate during training")
    ax2.set_xlabel("Time step")
    ax2.set_ylabel("Error rate")
    ax2.plot(validateError[1:])
    print("Training time without validation: %f " % training_time)
    # Make the prediction
    # for i in range(0,5):
    #     blk[i].set_final_parameters()
    blk.set_final_parameters()

    # Test on training set
    train_numPosInstance = 0
    train_numNegInstance = 0
    train_correctPos = 0
    train_correctNeg = 0
    with open("training_set.node_features") as traindata:
    # with open("training_set_noise_0.5.node_features") as traindata:
        while True:
            train_testline = traindata.readline().strip().split()
            train_testline = [float(x) for x in train_testline]
            if len(train_testline) == 0:
                break
            train_test_instance = Instance(train_testline)
            if train_test_instance.label == target1:
                train_numPosInstance += 1
            elif target2 == 0 and train_test_instance.label != target1:
                train_numNegInstance += 1
            elif target2 != 0 and train_test_instance.label == target2:
                train_numNegInstance += 1

            # Test
            train_test_mean, train_test_covariance = blk.prediction(train_test_instance)

            if target2 == 0 and train_test_mean >= 0 and train_test_instance.label == target1:
                train_correctPos += 1
            elif target2 == 0 and train_test_mean < 0 and train_test_instance.label != target1:
                train_correctNeg += 1
            elif target2 != 0 and train_test_mean >= 0 and train_test_instance.label == target1:
                train_correctPos += 1
            elif target2 != 0 and train_test_mean < 0 and train_test_instance.label == target2:
                train_correctNeg += 1
    train_confusion = np.array([[train_correctPos, train_numPosInstance - train_correctPos], [train_numNegInstance - train_correctNeg, train_correctNeg]])
    train_numInstance = train_numPosInstance + train_numNegInstance
    train_correct = train_correctPos + train_correctNeg
    train_accuracy = float(train_correct) / train_numInstance * 100
    print("The number of total instances in the two classes: %i" % train_numInstance)
    print("The number of correctlly classified: %i" % train_correct)
    print("The accuracy of classification: %.2f %%" % train_accuracy)
    print("The confusion matrix:")
    print train_confusion

    # Test on test set
    numPosInstance = 0
    numNegInstance = 0
    correctPos = 0
    correctNeg = 0
    plotPredictedVeg = np.zeros(3)
    plotPredictedWire = np.zeros(3)
    plotPredictedPole = np.zeros(3)
    plotPredictedGround = np.zeros(3)
    plotPredictedFacade = np.zeros(3)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlim(70, 280)
    ax.set_ylim(70, 280)
    ax.set_zlim(-10, 35)
    ax.set_title("Predicted labels for Veg and Facade")

    test_time = 0
    with open("test_set.node_features") as data:
    # with open("test_set_noise_0.5.node_features") as data:
        while True:
            testline = data.readline().strip().split()
            testline = [float(x) for x in testline]
            if len(testline) == 0:
                break
            test_instance = Instance(testline)
            if test_instance.label == target1:
                numPosInstance += 1
            elif target2 == 0 and test_instance.label != target1:
                numNegInstance += 1
            elif target2 != 0 and test_instance.label == target2:
                numNegInstance += 1

            # Test
            # test_mean = np.zeros(5)
            # test_covariance = np.zeros(5)
            # for i in range(0, 5):
            #     test_mean[i], test_covariance[i] = blk[i].prediction(test_instance)
            start_test = time.time()
            test_mean, test_covariance = blk.prediction(test_instance)
            stop_test = time.time()
            test_time += stop_test - start_test

            if target2 == 0 and test_mean >= 0 and test_instance.label == target1:
                correctPos += 1
            elif target2 == 0 and test_mean < 0 and test_instance.label != target1:
                correctNeg += 1
            elif target2 != 0 and test_mean >= 0 and test_instance.label == target1:
                correctPos += 1
            elif target2 != 0 and test_mean < 0 and test_instance.label == target2:
                correctNeg += 1

            if test_mean >= 0 and (test_instance.label == target1 or test_instance.label == target2):
                plotPredictedVeg = np.vstack((plotPredictedVeg, test_instance.pos))
            elif test_mean < 0 and (test_instance.label == target1 or test_instance.label == target2):
                plotPredictedFacade = np.vstack((plotPredictedFacade, test_instance.pos))

    print("Test time for all test data: %f " % test_time)
    # confusion matrix
    confusion = np.array([[correctPos, numPosInstance - correctPos], [numNegInstance - correctNeg, correctNeg]])
    numInstance = numPosInstance + numNegInstance
    correct = correctPos + correctNeg
    accuracy = float(correct) / numInstance * 100
    print("The number of total instances in the two classes: %i" % numInstance)
    print("The number of correctlly classified: %i" % correct)
    print("The accuracy of classification: %.2f %%" % accuracy)
    print("The confusion matrix:")
    print confusion

    if target2 == 0:
        plotClasses = ['plotPredictedVeg']
        color = ['g']
    else:
        plotClasses = ['plotPredictedVeg', 'plotPredictedFacade']
        color = ['g', 'y']
    # plotClasses = ['plotPredictedVeg', 'plotPredictedWire', 'plotPredictedPole', 'plotPredictedGround', 'plotPredictedFacade']
    # color = ['g', 'k', 'r', 'b', 'y']
    i = 0
    for name in plotClasses:
        temp = eval(name)
        ax.scatter(temp[1:, 0], temp[1:, 1], temp[1:, 2], s=5, c=color[i], depthshade=True, linewidth=0)
        i += 1

    plt.show()
if __name__ == "__main__":
    main()
