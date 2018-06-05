# -*- coding: utf-8 -*-
import sys, os
import random
import math
from math import exp
class neuron:
	def __init__(self, inputs, lamda):
		self.__inputs = inputs
		self.__weight = [] 
		self.__inpt = []
		self.__outpt = 0
		self.__net = 0 #base sum for i in  p(i)*w(i) for "inputs" number
		self.__lamda = lamda#learning rate
		self.setWeight(0, 0)
		self.setInput(0, 1)
		self.setWeights()
		#setter functions
	def setWeights(self):#initialize network weights by random nos
		for i in range(1, self.__inputs + 1):
			self.setWeight(i, random.random()-0.5)
	def setInput(self, index, inpt):#set i/p to perceptron
		if index < len(self.__inpt):
			self.__inpt[index] = inpt
		else:
			self.__inpt.append(inpt)
	def setOutput(self, outpt):#set output 
			self.__outpt = outpt
	def setNet(self, net=0):#set basefunction net(j) =sum over i for x(i)*w[i] 
		self.__net = net
	def getNumberInput(self):
		return self.__inputs
	def computeBaseFunction(self):
		self.setNet()
		for i in range(0, self.__inputs + 1):
			self.__net = self.__net + self.getWeight(i) * self.getInput(i)
	def computeOutput(self):
		y= 1/(1+exp(-self.__lamda*self.__net)) 
		self.setOutput(y)
	#bactrack utility
	def setWeight(self, index, weight):#set perticular perceptron weight by adding dw in w
		if index < len(self.__weight):
			self.__weight[index] = weight
		else:
			self.__weight.append(weight)
	def setDelta(self, delta):
		self.__delta = delta
	#getter functions :
	def getWeight(self, index):
		if index < len(self.__weight):
			return self.__weight[index]
	def getInput(self, index):
		if index < len(self.__inpt):
			return self.__inpt[index]
	def getOutput(self):
		return self.__outpt
	def getDelta(self):
		return self.__delta
	def getWs(self):
		return self.__weight
#Perceptron class is done 
#last modified 7 april 1:41 am

