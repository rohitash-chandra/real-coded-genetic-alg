# !/usr/bin/python

 
# Rohitash Chandra, Centre for Translational Data Science
# University of Sydey, Sydney NSW, Australia.  2018 c.rohitash@gmail.conm
# https://www.researchgate.net/profile/Rohitash_Chandra
# rohitash-chandra.github.

# Real coded genetic algoritm with Wright's heuristic crossover operator, roulette wheel selection, uniform mutation
#Problem - Rosenbrock function optimisation


import matplotlib.pyplot as plt
import numpy as np
import random
import time
import math 
import random  


class Evolution:
	def __init__(self, pop_size, num_variables, max_evals, max_limits, min_limits, xover_rate, mu_rate):
 
		self.pop   = [] 
		self.new_pop =  []

		self.pop_size = pop_size
		self.num_variables = num_variables 
		self.max_evals = max_evals 
		self.best_index = 0
		self.best_fit = 0
		self.worst_index = 0
		self.worst_fit = 0

		self.xover_rate = xover_rate 
		self.mu_rate = mu_rate 
		self.fit_list = np.zeros(pop_size)

		self.fit_ratio = np.zeros(pop_size)

		self.max_limits = max_limits # defines limits in your parameter space
		self.min_limits = min_limits 
		self.stepsize_vec  =  np.zeros(num_variables)
		self.step_ratio = 0.1 # determines the extent of noise you add when you mutate

		self.num_eval = 0

		self.problem = 1  # 1 rosen, 2 ellipsoidal 

		self.min_error = 1e-05
 


	def initialize(self):

		self.pop   =np.random.rand(self.pop_size, self.num_variables) 
		self.new_pop   = self.pop 

		span = np.subtract(self.max_limits,self.min_limits)


		for i in range(self.num_variables): # calculate the step size of each of the parameters
			self.stepsize_vec[i] = self.step_ratio  * span[i] 

	def print_pop(self ):

		print(self.pop, ' self.pop')
 


	def fit_func(self, x):    #   function  (can be any other function, model or even a neural network)
		fit = 0

		if self.problem == 1: # rosenbrock
			for j in range(x.size -1): 
				fit  = fit +  (100.0*(x[j]*x[j] - x[j+1])*(x[j]*x[j] - x[j+1]) + (x[j]-1.0)*(x[j]-1.0))

		elif self.problem ==2:  # ellipsoidal - sphere function
			for j in range(x.size):
				fit= fit + ((j+1)*(x[j]*x[j]))

		if fit ==0:
			fit = 1e-20
				  

		return 1/fit

	def evaluate_population(self): 

		self.fit_list[0] = self.fit_func(self.pop[0,:])
		self.best_fit = self.fit_list[0]
		self.best_index = 0

		sum = 0

		for i in range(self.pop_size):
			self.fit_list[i] = self.fit_func(self.pop[i,:]) 
			sum = sum + self.fit_list[i]

			if self.best_fit > self.fit_list[i]:
				self.best_fit = self.fit_list[i]
				self.best_index = i  

		self.num_eval = self.num_eval + self.pop_size

		print (sum, ' is sum ****** ')

		 
		for j in range(self.pop_size):
			self.fit_ratio[j] = (self.fit_list[j]/sum)* 100

 

	def roullete_wheel(self):

		wheel = np.zeros(self.pop_size+1) 
		wheel[0] = 0

		u = np.random.randint(100)

		if u == 0:
			u = 1  

		for i in range(1, wheel.size):
			wheel[i] = self.fit_ratio[i-1] + wheel[i-1] 
		#print(wheel, '  wheel') 
		#print (self.fit_ratio, ' self.fit_ratio') 
		#print(u, '    u ')  
		for j in range( wheel.size-1):
			if((u> wheel[j]) and (u < wheel[j+1])):
				return j   
	 
		return 0


	def xover_mutate(self, leftpair,rightpair):  # xover and mutate
 
		left = self.pop[leftpair,:]
		right = self.pop[rightpair,:]

		left_fit = self.fit_list[leftpair]
		right_fit =  self.fit_list[rightpair]

		#print(leftpair, rightpair, left_fit, right_fit, ' *      left right fit fit')

		u = random.uniform(0,1)

		if u < self.xover_rate:  # implement xover 

		 	alpha = random.uniform(0,1)

			if ( left_fit > right_fit): 
				child = (alpha *(left-right))+ left
		 	else: 
				child = (alpha *  (right-left))+ right   
		else: 
			child = left

		if u < self.mu_rate: # implement mutation 
			child =  np.random.normal(child, self.stepsize_vec)
 
			# check if the limits satisfy

		for j in range(child.size):
			if child[j] > self.max_limits[j]:
					child[j] = left[j]
			elif child[j] < self.min_limits[j]:
					child[j] = left[j] 

		return child  


	def evo_alg(self):

		self.initialize()

		self.print_pop()

		global_bestfit = 0
		global_best = []
		global_bestindex = 0
 

		while(self.num_eval< self.max_evals):

			self.evaluate_population() 

 
			for i in range(1, self.pop_size):
 
				leftpair =  self.roullete_wheel() #np.random.randint(self.pop_size) 
				rightpair = self.roullete_wheel()  # np.random.randint(self.pop_size)  

				while (leftpair == rightpair): 
					leftpair =  self.roullete_wheel()  # np.random.randint(self.pop_size) 
					rightpair = self.roullete_wheel()  # np.random.randint(self.pop_size)  
	   
				self.new_pop[i,:] = self.xover_mutate(leftpair,rightpair)


			best = self.pop[self.best_index, :]

			if self.best_fit > global_bestfit:
				global_bestfit = self.best_fit 
				global_best = self.pop[self.best_index, :]

			print(global_bestfit, ' global_best') 

			print(global_best, ' global_best sol')


			self.pop = self.new_pop


			self.pop[0,:] = global_best # ensure that you retain the best 

 

			print(self.num_eval, self.best_fit, ' local best fit so far')
			print(best, 'local best so far') 
 
		 	if  (1/self.best_fit) < self.min_error:
		 		print(' reached min error')
				break 
 
 

		self.print_pop() 

		return best, 1/self.best_fit, global_best, 1/global_bestfit

 

def main():


	random.seed(time.time()) 

	min_fitness = 0.005  # stop when fitness reaches this value. not implemented - can be implemented later


	max_evals = 20000   # need to decide yourself 80000

	pop_size = 50
	num_variables = 10 

	xover_rate = 0.8
	mu_rate = 0.1

	#max_limits = [2, 2, 2, 2, 2]
	max_limits = np.repeat(2, num_variables)
	#min_limits = [0, 0, 0, 0, 0] 

	min_limits = np.repeat(0, num_variables)
 

	evo = Evolution(pop_size, num_variables, max_evals,  max_limits, min_limits, xover_rate, mu_rate)


	best, best_fit, global_best, global_bestfit = evo.evo_alg()


	print(best, ' retruned best ')

	print(best_fit, ' retruned best fit')


	print(global_best, ' retruned global best ')

	print(global_bestfit, ' retruned global bestfit')
 
 



 

if __name__ == "__main__": main()
