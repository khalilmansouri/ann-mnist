
import network as NN
import dataLoader as loader


trainingData = loader.training_data()
testingData = loader.testing_data()

net = NN.Network([784,100,30, 10])
steps = 30

net.StochasticGradientDescent(trainingData, steps, test_data=testingData)
net.save('trainedNet')
