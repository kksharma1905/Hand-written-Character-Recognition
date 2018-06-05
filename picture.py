#class for image processing 
#picture need to be preprocessed before given to the neural network as input as it contains noise elements 
from PIL import Image
import sys, os
class Pic:
	def __init__(self, path):#path of image in folders
		self.path = path#path of picture in system
		self.threshold = 128.0#greyscale pixel values above it ==1 and below it ==0  
		self.vector = []
	def getVector(self):
		#image = image.resize((10,10), Image.ANTIALIAS)
		#above method is best method to resize an image
		try:
			image = Image.open(self.path)#image library has inbuilt classes for image rendering and i/o
		except:
			sys.stderr.write("ERROR: path %s is invalid ( NOT FOUND )\n" % self.path)#exception 
			sys.exit()
		pixel_list = list(image.getdata()) # store rgb data of image in list
		for i in pixel_list:
			greyscale = 0.3*i[0] + 0.6*i[1] + 0.1*i[2]#converting into greyscale
			if (self.threshold < greyscale):#sort of edge detection and noise cancelling  
				self.vector.append(float(0.0))#white
			else:
				self.vector.append(float(1.0))#black
		return self.vector
#Pic class is done
#last modified 7 april 12:41 am
