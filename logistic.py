import csv 
import numpy as np 
import matplotlib.pyplot as plt 

def loadCSV(filename): 
	with open(filename,"r") as csvfile: 
		lines = csv.reader(csvfile) 
		dataset = list(lines) 
		for i in range(len(dataset)): 
			dataset[i] = [float(x) for x in dataset[i]]	 
	return np.array(dataset)


def normalize(X): #normalize feature
	mins = np.min(X, axis = 0) 
	maxs = np.max(X, axis = 0) 
	range = maxs - mins 
	norm_X = 1 - ((maxs - X)/range) 
	return norm_X 

def log_fn(b, X): #log fn
	return 1.0/(1 + np.exp(-np.dot(X, b.T))) 

def log_gradient(b, X, y): #log gradient fn
	first_cal = log_fn(b, X) - y.reshape(X.shape[0], -1) 
	final_cal = np.dot(first_cal.T, X) 
	return final_cal 

def cost_fn(b, X, y): #cost fn
	log_fn_v = log_fn(b, X) 
	y = np.squeeze(y) 
	step1 = y * np.log(log_fn_v) 
	step2 = (1 - y) * np.log(1 - log_fn_v) 
	final = -step1 - step2 
	return np.mean(final) 

def grad_desc(X, y, b, lr=.01, converge_change=.001):
	cost = cost_fn(b, X, y) 
	change_cost = 1
	num_iter = 1
	while(change_cost > converge_change): 
		old_cost = cost 
		b = b - (lr * log_gradient(b, X, y)) 
		cost = cost_fn(b, X, y) 
		change_cost = old_cost - cost 
		num_iter += 1	
	return b, num_iter 


def pred_values(b, X): #predict labels b-->beta
	pred_prob = log_fn(b, X) 
	pred_value = np.where(pred_prob >= .5, 1, 0) 
	return np.squeeze(pred_value) 


def plot_reg(X, y, b): #plot decision boundary
	# labelled observations 
	x_0 = X[np.where(y == 0.0)] 
	x_1 = X[np.where(y == 1.0)]  
	plt.scatter([x_0[:, 1]], [x_0[:, 2]], c='red', label='y = 0') 
	plt.scatter([x_1[:, 1]], [x_1[:, 2]], c='green', label='y = 1') 
	#decision boundary 
	x1 = np.arange(0, 1, 0.1) 
	x2 = -(b[0,0] + b[0,1]*x1)/b[0,2] 
	plt.plot(x1, x2, c='black', label='reg line') 
	plt.xlabel('x1') 
	plt.ylabel('x2') 
	plt.legend() 
	plt.show() 
		
if __name__ == "__main__": 
	dataset = loadCSV('dataset.csv') 
	X = normalize(dataset[:, :-1]) 
	X = np.hstack((np.matrix(np.ones(X.shape[0])).T, X)) 
	y = dataset[:, -1]
	#initial b values 
	b = np.matrix(np.zeros(X.shape[1])) 
	#b values after gradient descent 
	b, num_iter = grad_desc(X, y, b) 
	print("Est.coefficients:", b) 
	print("No. of iterations:", num_iter) 
	y_pred = pred_values(b, X) 
	print("Correctly predicted labels:", np.sum(y == y_pred))  
	plot_reg(X, y, b) 
