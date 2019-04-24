import os
import argparse
import numpy as np
from preprocessor import Preprocessor
from network import NeuralNetwork

def test(trainDataFile, testDataFile, attrDataFile, activationFunc):
    data = Preprocessor(trainDataFile, testDataFile, attrDataFile)
    act = activationFunc
    data.loadData()
    trainData = data.getMatrix(data.getTrainData())
    testData = data.getMatrix(data.getTestData())
 
    numInput = data.getNumInput()
    numOutput = len(data.getClasses())
    numHidden = 3
    seed = 4 
    learningRate = 0.1
    maxEpochs = 1000
    momentum = 0.0

    print("Generating neural network: %d-%d-%d" % (numInput, numHidden,numOutput))
    nn = NeuralNetwork(numInput, numHidden, numOutput, seed, activation=act)
    nn.train(trainData, maxEpochs, learningRate, momentum)
    print("Training complete")

 #   accTrain = nn.accuracy(trainData)
    accTest = nn.accuracy(testData)

 #   print("\nAccuracy on train data = %0.4f " % accTrain)   
    print("Accuracy on test data   = %0.4f " % accTest)
  


def main():
    parser = argparse.ArgumentParser(description="Decision Tree")
    parser.add_argument("-e", "--experiment", required=True, dest="experiment", 
            choices=["testIris", "testWine", "testAdult"], help="experiment name.")
    parser.add_argument("-a", "--activation", required=False, default="comb", dest="activation",
            choices=["comb", "sigmoid", "tanh", "leakyRelu"], help='activation function')
    args = parser.parse_args()
    print("Experiment: " + args.experiment)
    print("Activation Function: " + args.activation)
    if args.experiment == "testIris":
        test("data/iris/iris-train.txt", "data/iris/iris-test.txt", "data/iris/iris-attr.txt", args.activation)
    elif args.experiment == "testWine":
        test("data/wine/wine-train.txt", "data/wine/wine-test.txt", "data/wine/wine-attr.txt", args.activation)
    elif args.experiment == "testAdult":
        test("data/adult/adult-train.txt", "data/adult/adult-test.txt", "data/adult/adult-attr.txt", args.activation)

if __name__ == '__main__':
    main()
