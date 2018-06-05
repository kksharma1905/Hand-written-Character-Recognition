# -*- coding: utf-8 -*-
import math
import sys, os
import picture
import neuron

firstLayer = 25
hiddenLayer = 20
lastLayer = 10
lamda = 0.5
directory = "./training_set/"	#training directory
directoryTest = "./testing_set/"	#testing directory
inputNeurons = [] #stores input Neurons = 8 
hiddenNeurons = []#strores hidden Neurons =12 
lastNeurons = []#stores o/p Neurons = 10
imageVectors = {}
#########################   build and train network   ##########################
def build_N_learn(lamda):
	tset = os.listdir(directory)
	for n in tset:
		p = picture.Pic(os.path.join(directory, n))
		filename = n.split(".")
		imageVectors[filename[0]] = p.getVector()
	TotalErr = 0.0
	Err = 0
	iteration = 0
	while (Err == 0.0 or TotalErr > 2 and iteration < 2500):
		if iteration % 20 == 0: 
			print "total error till " , iteration ,"th iteration is :", TotalErr

		iteration = iteration+1
		TotalErr = 0

		for k in imageVectors.keys():
			for i in range(firstLayer):
				if len(inputNeurons) < firstLayer:
					f = neuron.neuron(len(imageVectors[k]), lamda)
					f.setWeights()
					inputNeurons.append(f)
				for y in range(len(imageVectors[k])):
					inputNeurons[i].setInput(y+1, imageVectors[k][y])
				inputNeurons[i].computeBaseFunction()
				inputNeurons[i].computeOutput()
			#print k
			for i in range(hiddenLayer):
				if len(hiddenNeurons) < hiddenLayer:
					h = neuron.neuron(firstLayer, lamda)
					h.setWeights()
					hiddenNeurons.append(h)
				for y in range(firstLayer):
					hiddenNeurons[i].setInput(y+1, inputNeurons[y].getOutput())
				hiddenNeurons[i].computeBaseFunction()
				hiddenNeurons[i].computeOutput()

			for i in range(lastLayer):
				if len(lastNeurons) < lastLayer:
					l = neuron.neuron(hiddenLayer, lamda)
					l.setWeights()
					lastNeurons.append(l)
				for y in range(hiddenLayer):
					lastNeurons[i].setInput(y+1, hiddenNeurons[y].getOutput())
				lastNeurons[i].computeBaseFunction()
				lastNeurons[i].computeOutput()

			Err = 0
			for i in range(lastLayer):#delta calculation for o/p layer
				if i == int(k[0]):#check for which o/p neuron should give 1 as o/p
					o=lastNeurons[i].getOutput()
					Err = Err + (1 - o)**2
					lastNeurons[i].setDelta( (1 - o) * lamda * o * (1 - o) )
				else:
					o=lastNeurons[i].getOutput()
					Err = Err + (0 - lastNeurons[i].getOutput())**2
					lastNeurons[i].setDelta( (0 - o) * lamda * o * (1 - o) )
			TotalErr = TotalErr + 0.5 * Err

			####Backpropagation
			for i in range(hiddenLayer):
				sum = 0
				for sd in range(lastLayer):
					sum = sum + lastNeurons[sd].getDelta() * lastNeurons[sd].getWeight(i+1)
				#delta calc for hidden layer propagated from o/p layer
				o=hiddenNeurons[i].getOutput()
				hiddenNeurons[i].setDelta(sum * lamda * o * (1 - o))

			for i in range(firstLayer):
				sum = 0
				for sd in range(hiddenLayer):
					sum = sum+ hiddenNeurons[sd].getDelta() * hiddenNeurons[sd].getWeight(i+1)
				#delta calc for hidden layer propagated from prev hidden  layer
				o=inputNeurons[i].getOutput()
				inputNeurons[i].setDelta(sum * lamda * o * (1 - o))

			for i in range(0, firstLayer):
				for j in range(0, inputNeurons[i].getNumberInput()):
					nw=inputNeurons[i].getWeight(j) + 0.6 * inputNeurons[i].getDelta() * inputNeurons[i].getInput(j)
					inputNeurons[i].setWeight(j, nw)

			for i in range(0, hiddenLayer):
				for j in range(0, hiddenNeurons[i].getNumberInput()):
					nw=hiddenNeurons[i].getWeight(j) + 0.6 * hiddenNeurons[i].getDelta() * hiddenNeurons[i].getInput(j)
					hiddenNeurons[i].setWeight(j, nw)

			for i in range(0, lastLayer):
				for j in range(0, lastNeurons[i].getNumberInput()):
					nw=lastNeurons[i].getWeight(j) + 0.6 * lastNeurons[i].getDelta() * lastNeurons[i].getInput(j)
					lastNeurons[i].setWeight(j, nw)

####################################### testing ########################################
def test_NN():
	imageVectors = {}
	t=10
	c=0
	print "Test is going on :"
	tset = os.listdir(directoryTest)
	for n in tset:
		p = picture.Pic(os.path.join(directoryTest, n))
		filename = n.split(".")
		imageVectors[filename[0]] = p.getVector()
	for i in range(firstLayer):
		print inputNeurons[i].getWs()
	for k in imageVectors.keys():
		for i in range(firstLayer):
			for y in range(len(imageVectors[k])):
				inputNeurons[i].setInput(y+1, imageVectors[k][y])
			inputNeurons[i].computeBaseFunction()
			inputNeurons[i].computeOutput()

		for i in range(hiddenLayer):
			for y in range(firstLayer):
				hiddenNeurons[i].setInput(y+1, inputNeurons[y].getOutput())
			hiddenNeurons[i].computeBaseFunction()
			hiddenNeurons[i].computeOutput()

		for i in range(lastLayer):
			for y in range(hiddenLayer):
				lastNeurons[i].setInput(y+1, hiddenNeurons[y].getOutput())
			lastNeurons[i].computeBaseFunction()
			lastNeurons[i].computeOutput()

		max = 0.0
		temp=0
		for y in range(lastLayer):
			if max < lastNeurons[y].getOutput():
				max = lastNeurons[y].getOutput()
				temp = y

		for y in range(lastLayer):
			if int(k[0]) == y:
				if max == lastNeurons[y].getOutput():
					print "for character ",int(k[0]), "result is correct with output at ",y,"th neuron=", lastNeurons[y].getOutput()
					c=c+1
				else:
					print "for character ",int(k[0]), "result is incorrect with as", lastNeurons[y].getOutput()," much near to ",temp," with o/p of ",lastNeurons[temp].getOutput()
	print "out of", t ,"total correct identification are ",c
#operate neural network
build_N_learn(lamda)
print "input neurons :",firstLayer
print "hidden neurons :",hiddenLayer
print "output neurons :",lastLayer
test_NN()
