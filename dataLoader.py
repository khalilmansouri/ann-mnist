import numpy as np
from array import array as array
import struct 

# training data location
trainingImagePath = './data/train/train-images-idx3-ubyte'
trainingLabelPath = './data/train/train-labels-idx1-ubyte'

# test data
testImagePath = './data/test/t10k-images-idx3-ubyte'
testLabelPath = './data/test/t10k-labels-idx1-ubyte'

# number of pixels in each image
imageSize = 784



#load training images into 2D table
def loadImage(path):
    with open(path,'rb') as f:
        magicNumber, size = struct.unpack(">II", f.read(8)) # {1,2,3,4} = magic, {5,6,7,8} = size
        nrows, ncols = struct.unpack(">II", f.read(8))
        data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        #data = data.reshape((size*imageSize, 1))
        data = data.reshape((size, imageSize))
        fdata = [np.reshape(image,(-1,1 ))/225. for image in data]
        return fdata
    

#load taining labels into 1D table
def loadLabels(path):
    with open(path,'rb') as f:
        #struct.unpack(">II", f.read(8))
        f.read(8)
        data = array("B", f.read())
        return [np.reshape(vectorize(label), (-1, 1)) for label in data]
        
#load test labels into 1D table
def loadTestLabels(path):
    with open(path,'rb') as f:
        #struct.unpack(">II", f.read(8))
        f.read(8)
        data = array("B", f.read())
        return data
        #return [np.reshape(label,(-1,1)) for label in data]


def vectorize(label):
    output = np.zeros(10)
    output[label] = 1.0
    return output

# normalize function 
def normalize(pixel):
    return float(pixel / 255)
normalize_v = np.vectorize(normalize)


def training_data():
    return zip(loadImage(trainingImagePath), loadLabels(trainingLabelPath))

def testing_data():
    return zip(loadImage(testImagePath), loadTestLabels(testLabelPath))