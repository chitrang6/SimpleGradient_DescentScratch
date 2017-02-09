# THis is the simple gradient descent algorithm implementation in python using numpy


import numpy as np

# Find the error or loss function for the simple linear regression 

def errorFunctionLineraRegression(dataset , biasval , xfeatweightval):
	#This function will compute the Sum of the squared error. 

	N = float(len(dataset))
	SumError  = 0.0
	for i in xrange(len(dataset)):
		X = dataset[i , 0]
		y = dataset[i , 1]

		SumError += (y - (biasval + X * xfeatweightval)) ** 2
	return SumError / N






def gradient_decent_algorithm(dataset, learning_rate, numofIterations, bias, Xweight):

	# Now, We will try to minimize the above loss function meaning try to minimize the sum of squared errors.
	cur_bias = bias 
	cur_Xweight = Xweight
	for i in xrange(numofIterations):
		Finalbias, FinalXweight = update_parameters_step(dataset, learning_rate, cur_bias, cur_Xweight)

	return [Finalbias , FinalXweight]




def update_parameters_step(dataset , learning_rate , b , W):
	# Here we will go through the data set and will update the bias and weight according to the steepest gradient direction 
	# Meaning we are going to find the partial derivatives
	b_gradient = 0.0
	w_gradient = 0.0
	N = float(len(dataset))
	for i in xrange(len(dataset)):
		X = dataset[i , 0]
		y = dataset[i , 1]
		b_gradient  += -(2/N) * (y - ((W * X) + b))
		w_gradient += -(2/N) * X * (y - ((W * X) + b))

	update_b = b - b_gradient * learning_rate
	update_w = W - w_gradient * learning_rate

	return [update_b , update_w]






def linear_regression():
	# First task is to load the data set.
	dataset = np.genfromtxt('data.csv' , delimiter = ',')

	# In this numpy array we will have our data set which has basically two cols one is for x and one is for y.
	# Our task is to find the best line which fits this data-set with minimum error.
	# To help out us with tis task we are going to use gradient descent algortihm to find the best fitting line for this data-set.


	# Hyper parameters 
	learning_rate = 0.0001
	numofIterations = 10000
	initial_bias = 0
	initial_Xweight = 0


	print "Starting point of finding the best fit --> bias - %f  X Feature weight - %f  Initial Error - %f" % (initial_bias , initial_Xweight ,
	 errorFunctionLineraRegression(np.array(dataset), initial_bias , initial_Xweight))

	[final_bias , final_XFeat]= gradient_decent_algorithm(np.array(dataset), learning_rate, numofIterations, initial_bias, initial_Xweight)

	print "Ending point of finding the best fit --> bias - %f  X Feature weight - %f  Final Error - %f" % (final_bias , final_XFeat ,
	 errorFunctionLineraRegression(np.array(dataset), final_bias , final_XFeat))



if __name__ == '__main__':
	linear_regression()


