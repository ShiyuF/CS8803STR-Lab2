#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import csv
from mpl_toolkits.mplot3d import Axes3D
import time
import linecache

x_veg, y_veg, z_veg  = [[] for i in range(3)]
x_wire, y_wire, z_wire  = [[] for i in range(3)]
x_pole, y_pole, z_pole  = [[] for i in range(3)]
x_ground, y_ground, z_ground  = [[] for i in range(3)]
x_facade, y_facade, z_facade  = [[] for i in range(3)]

node_id = []
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d', title='Actual data (Truth)')

def plot_scene():
	with open('combineddata','rb') as datafile:
		reader = csv.reader(datafile)
		for row in reader:
				cur_row = row[0].split(' ')
				node_id.append(int(cur_row[4]))
		
				if node_id[len(node_id)-1] == 1004:
					x_veg.append(float(cur_row[0]))
					y_veg.append(float(cur_row[1]))
					z_veg.append(float(cur_row[2]))
				elif node_id[len(node_id)-1] == 1100:
					x_wire.append(float(cur_row[0]))
					y_wire.append(float(cur_row[1]))
					z_wire.append(float(cur_row[2]))
				elif node_id[len(node_id)-1] == 1103:
					x_pole.append(float(cur_row[0]))
					y_pole.append(float(cur_row[1]))
					z_pole.append(float(cur_row[2]))			
				elif node_id[len(node_id)-1] == 1200:
					x_ground.append(float(cur_row[0]))
					y_ground.append(float(cur_row[1]))
					z_ground.append(float(cur_row[2]))				
				elif node_id[len(node_id)-1] == 1400:
					x_facade.append(float(cur_row[0]))
					y_facade.append(float(cur_row[1]))
					z_facade.append(float(cur_row[2]))
		m = 0.55			
		ax.plot(x_veg, y_veg, z_veg, c='darkgreen', marker='.', markeredgecolor="darkgreen", linestyle='None', markersize=m)
		ax.plot(x_wire, y_wire, z_wire, c='black', marker='.', markeredgecolor="black", linestyle='None', markersize=m)
		ax.plot(x_pole, y_pole, z_pole, c='silver', marker='.', markeredgecolor="silver", linestyle='None', markersize=m)
		ax.plot(x_ground, y_ground, z_ground, c='sandybrown', marker='.', markeredgecolor="sandybrown", linestyle='None', markersize=m)
		ax.plot(x_facade, y_facade, z_facade, c='yellow', marker='.', markeredgecolor="yellow", linestyle='None', markersize=m)
		ax.axis([80, 250, 100, 260])
		ax.view_init(40,155)
		
		'''
		ax.scatter(x_veg,y_veg,z_veg, c='darkgreen', marker=',', s=0.5, edgecolor="darkgreen")
		ax.scatter(x_wire,y_wire,z_wire,c='black', marker=',', s=0.5, edgecolor="black")
		ax.scatter(x_pole,y_pole,z_pole,c='silver', marker=',', s=0.5, edgecolor="silver")
		ax.scatter(x_ground,y_ground,z_ground, c='sandybrown', marker=',', s=0.5, edgecolor="sandybrown")
		ax.scatter(x_facade,y_facade,z_facade, c='yellow', marker=',', s=0.5, edgecolor="yellow")
		plt.show()
		'''

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
	
	x_result_facade, y_result_facade, z_result_facade  = [[] for i in range(3)]
	x_result_veg, y_result_veg, z_result_veg  = [[] for i in range(3)]
				
	num_lines = open('test_set').read().count('\n')
	tp = 0.0
	tn = 0.0
	fp = 0.0
	fn = 0.0
	for i in range(num_lines):
		line = linecache.getline('test_set',i+1)
		vals = [float(j) for j in line.split()]
		feature = vals[5:15]
		f_vec = np.asarray(feature,dtype=np.float32)
		node_id = int(vals[4])
		if node_id == 1004 or node_id == 1400:
			result = np.dot(wts,f_vec)
			epsilon = 0.0
			if result >= epsilon and node_id == 1004: 
				tp = tp + 1
				x_result_veg.append(float(vals[0])) 
				y_result_veg.append(float(vals[1])) 
				z_result_veg.append(float(vals[2]))
			elif result >= epsilon and node_id == 1400: fp = fp + 1
			elif result < epsilon and node_id == 1004: fn = fn + 1
			elif result < epsilon and node_id == 1400 : 
				tn = tn +1
				x_result_facade.append(float(vals[0])) 
				y_result_facade.append(float(vals[1])) 
				z_result_facade.append(float(vals[2]))
	print ("tp = %d, tn = %d, fp  = %d, fn = %d" % (tp,tn,fp,fn))	
	print ("Accuracy=%f" % (float((tp+tn))/(tp+tn+fp+fn)))
				
	ax2 = fig.add_subplot(122, projection='3d', title='SVM on Test data set for Ground & Veg')
	ax2.plot(x_result_veg, y_result_veg, z_result_veg, c='darkgreen', marker='.', markeredgecolor="darkgreen", linestyle='None', markersize=0.5)
	ax2.plot(x_result_facade, y_result_facade, z_result_facade, c='yellow', marker='.', markeredgecolor="yellow", linestyle='None', markersize=0.5)
	ax2.axis([80, 250, 100, 260])
	ax2.view_init(40,155)
	plt.show()
	
if __name__ == '__main__':
	plot_scene()
	time.sleep(2)
	train_svm()
