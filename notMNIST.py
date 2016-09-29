from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D
import pickle
import numpy as np
from random import randint

model = Sequential()
#model.add(Convolution2D(16,3, border_mode="same", input_shape=(28,28)))
model.add(Dense(64,activation="relu", input_dim=784))
model.add(Dense(16,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

#create training set
import string

data = pickle.load(open("notMNIST_sets.pickle","rb"))

pretrainSet = [x[0] for x in data["train"][:100000]]
pretrainLabels = [x[1] for x in data["train"][:100000]]
trainSet = np.ndarray(shape=(100000,784),dtype=np.float32)
trainLabels = np.ndarray(shape=(100000,10),dtype=np.float32)

ab = ['A','B','C','D','E','F','G','H','I','J']
oneHotLabels = [[(1 if ab[x] == y else 0) for x in range(0, len(ab))] for y in pretrainLabels]

pretestSet = [x[0] for x in data["test"]][:10000]
pretestLabels = [x[1] for x in data["test"]][:10000]
testSet = np.ndarray(shape=(10000,784),dtype=np.float32)
testLabels = np.ndarray(shape=(10000,10),dtype=np.float32)

oneHotTest = [[(1 if ab[x] == y else 0) for x in range(0, len(ab))] for y in pretestLabels]

for x in range(0,100000):
	trainSet[x] = pretrainSet[x].flatten()
	trainLabels = oneHotLabels[x]

for x in range(0,10000):
	testSet[x] = pretestSet[x].flatten()


#labeledTrain = list(zip(trainData,trainLabels))
#shuffled = [labeledTrain[randint(0,99999)] for x in range(0,50000)]


#trainData,trainLabels = zip(*labeledTrain)
#trainData = list(trainData)

#trainDataFinal = trainData.reshape(100000,784)
model.fit(trainSet, oneHotLabels, nb_epoch=20, batch_size = 250, verbose=1)
print(model.test_on_batch(testSet, oneHotTest))
