#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import csv
from mpl_toolkits.mplot3d import Axes3D
import time
import linecache

def train_svm():
	'''
	Picking Veg(1004) and Facade (1400) as the two classes
	Let Veg have label 1
	Let Facade have label -1
	'''
	
	lam = 0.495
	num_lines = open('training_set').read().count('\n')
	wts = 2 * np.random.uniform(0, 1, 10) - 1
	c = 0
	sse = []
	itr = []
	for i in range(num_lines):
		line = linecache.getline('training_set',i+1)
		vals = [float(j) for j in line.split()]
		feature = vals[5:15]
		f_vec = np.asarray(feature,dtype=np.float32)
		node_id = int(vals[4])
		if node_id == 1004 or node_id == 1400:
			c = c + 1
			alpha_t = 0.01/np.sqrt(float(c))
			model_value = np.dot(wts, f_vec)
			if node_id == 1004: y = 1
			else : y = -1
			eps = 0.0
			if model_value >= eps and node_id == 1004: # true positive
				wts = wts - 2*alpha_t*lam*wts
			elif model_value >= eps and node_id == 1400: # false positive
				wts = wts - 2*alpha_t*lam*wts + alpha_t*y*f_vec 
			elif model_value < eps and node_id == 1004: # false negative
				wts = wts - 2*alpha_t*lam*wts + alpha_t*y*f_vec
			elif model_value < eps and node_id == 1400: # true negative
				wts = wts - 2*alpha_t*lam*wts
			
			if c == 1:
				diff = 0
				itr.append(c)
				num_lines = open('test_set').read().count('\n')
				for i in range(num_lines):
					line = linecache.getline('test_set',i+1)
					vals = [float(j) for j in line.split()]
					feature = vals[5:15]
					f_vec = np.asarray(feature,dtype=np.float32)
					node_id = int(vals[4])
					if node_id == 1004 or node_id == 1400:
						result = np.dot(wts,f_vec)
						if node_id == 1004: y = 1
						else: y = -1
						diff = diff + (y - result)**2
				sse.append(diff)
					
			if c%1000 == 0:
				diff = 0
				itr.append(c)
				num_lines = open('test_set').read().count('\n')
				for i in range(num_lines):
					line = linecache.getline('test_set',i+1)
					vals = [float(j) for j in line.split()]
					feature = vals[5:15]
					f_vec = np.asarray(feature,dtype=np.float32)
					node_id = int(vals[4])
					if node_id == 1004 or node_id == 1400:
						result = np.dot(wts,f_vec)
						if node_id == 1004: y = 1
						else: y = -1
						diff = diff + (y - result)**2
				sse.append(diff)
	
	plt.plot(itr, sse, 'r--')
	plt.xlabel('Iteration Number')
	plt.ylabel('Sum Squared Error')
	plt.title('Error Convergence plot')
	plt.grid(True)
	plt.show()
				
				
if __name__ == '__main__':
	train_svm()

