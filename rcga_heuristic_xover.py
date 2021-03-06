# !/usr/bin/python

 
# Rohitash Chandra, Centre for Translational Data Science
# University of Sydey, Sydney NSW, Australia.  2018 c.rohitash@gmail.conm
# https://www.researchgate.net/profile/Rohitash_Chandra
# rohitash-chandra.github.

# Real coded genetic algoritm with Wright's heuristic crossover operator, roulette wheel selection, uniform mutation
#Problem - Rosenbrock function optimisation


# Source: Wright, A. H. (1991). Genetic algorithms for real parameter optimization. In Foundations of genetic algorithms (Vol. 1, pp. 205-218). Elsevier.
#https://www.sciencedirect.com/science/article/pii/B9780080506845500161
#http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.25.5297&rep=rep1&type=pdf

#Goldberg, David E., and Kalyanmoy Deb. "A comparative analysis of selection schemes used in genetic algorithms." Foundations of genetic algorithms. Vol. 1. Elsevier, 1991. 69-93.


#[Eshelman, Larry J., and J. David Schaffer. "Real-coded genetic algorithms and interval-schemata." Foundations of genetic algorithms. Vol. 2. Elsevier, 1993. 187-202.](https://www.sciencedirect.com/science/article/pii/B9780080948324500180)
#[download](http://heuristic.kaist.ac.kr/cylee/xga/Paper%20Review/Papers/[P6]%20Real-Coded%20Genetic%20Algorithms.pdf)

#[Blanco, Armando, Miguel Delgado, and Maria C. Pegalajar. "A real-coded genetic algorithm for training recurrent neural networks." Neural networks 14.1 (2001): 93-105.](https://www.sciencedirect.com/science/article/pii/S0893608000000812)

import matplotlib.pyplot as plt
import numpy as np
import random
import time
import math 
import random  


class Evolution:
	def __init__(self, prob, pop_size, num_variables, max_evals, max_limits, min_limits, xover_rate, mu_rate, min_fitness, chose_xover):
 
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

		self.num_eval = 0 # begin with

		self.problem = prob  # 1 rosen, 2 ellipsoidal 

		self.min_error = min_fitness

		self.xover = chose_xover  # 1. Wrights Heuristic Xover, 2. Simulated Binary Xover
 


	def initialize(self):

		self.pop   =np.random.rand(self.pop_size, self.num_variables) 
		self.new_pop   = self.pop 

		span = np.subtract(self.max_limits,self.min_limits)


		for i in range(self.num_variables): # calculate the step size of each of the parameters. just for mutation by random-walk
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
			fit = 1e-20 # to be safe with division by 0
				  

		return 1/fit # note we will maximize fitness, hence minimize error 

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
		for j in range( wheel.size-1):
			if((u> wheel[j]) and (u < wheel[j+1])):
				return j   
	 
		return 0

 

	def sim_binary_xover(self,ind1, ind2): # Simulated Binary Crossover https://github.com/DEAP/deap/blob/master/deap/tools/crossover.py

	#:param ind1: The first individual participating in the crossover.
	#:param ind2: The second individual participating in the crossover.
	#:param eta: Crowding degree of the crossover. A high eta will produce
	 #           children resembling to their parents, while a small eta will
	  #          produce solutions much more different.
		x1 = ind1
		x2  = ind2 

		eta = 2

		for i, (x1, x2) in enumerate(zip(ind1, ind2)):
			rand = random.random()
			if rand <= 0.5:
				beta = 2. * rand
			else:
				beta = 1. / (2. * (1. - rand))

			beta **= 1. / (eta + 1.)

			ind1[i] = 0.5 * (((1 + beta) * x1) + ((1 - beta) * x2))
			ind2[i] = 0.5 * (((1 - beta) * x1) + ((1 + beta) * x2))


		return ind1, ind2

 


	def blend_xover(self,ind1, ind2):
		'''Executes a blend crossover that modify in-place the input individuals. 
		:param ind1: The first individual participating in the crossover.
		:param ind2: The second individual participating in the crossover.
		:param alpha: Extent of the interval in which the new values can be drawn
				  for each attribute on both side of the parents' attributes. '''
		alpha = 0.1
		x1 = ind1
		x2 = ind2


		for i in range(ind1.size): #(x1, x2) in enumerate(zip(ind1, ind2)):
			gamma = (1. + 2. * alpha) * random.random() - alpha
			ind1[i] = (1. - gamma) * x1[i] + gamma * x2[i]
			ind2[i] = gamma * x1[i] + (1. - gamma) * x2[i]

		return ind1, ind2




	def wrights_xover(self,left_fit, right_fit, left, right):   #Wrights Heuristic Crossover

		alpha = random.uniform(0,1) 
		if ( left_fit > right_fit): 
			child = (alpha *(left-right))+ left
		else: 
			child = (alpha *  (right-left))+ right

		return child

	def impose_limits(self, child, prev):

		for j in range(child.size):
			if child[j] > self.max_limits[j]:
					child[j] = prev[j]
			elif child[j] < self.min_limits[j]:
					child[j] = prev[j] 
		return child 

	def xover_mutate(self, leftpair,rightpair):  # xover and mutate
 
		left = self.pop[leftpair,:]
		right = self.pop[rightpair,:]

		left_fit = self.fit_list[leftpair]
		right_fit =  self.fit_list[rightpair] 

		u = random.uniform(0,1)

		if u < self.xover_rate:  # implement xover  

			if self.xover == 1:
				child_one = self.wrights_xover(left_fit, right_fit, left, right ) 
				child_two = self.wrights_xover(left_fit, right_fit, left, right ) 

			elif self.xover == 2:
				child_one, child_two = self.sim_binary_xover( left, right)

			elif self.xover == 3:
				child_one, child_two = self.blend_xover( left, right)

		else: 
			child_one = left
			child_two = right

		if u < self.mu_rate: # implement mutation 
			child_one =  np.random.normal(child_one, self.stepsize_vec) 
			child_two =  np.random.normal(child_two, self.stepsize_vec)
 
			# check if the limits satisfy - keep in range 
		child_one = self.impose_limits(child_one, left) 
		child_two = self.impose_limits(child_two, right)

		return child_one, child_two


	def evo_alg(self):

		self.initialize()
 
		global_bestfit = 0
		global_best = []
		global_bestindex = 0
 

		while(self.num_eval< self.max_evals):

			self.evaluate_population() 

 
			for i in range(0,  self.pop_size, 2):
 
				leftpair =  self.roullete_wheel() #np.random.randint(self.pop_size) 
				rightpair = self.roullete_wheel()  # np.random.randint(self.pop_size)  

				while (leftpair == rightpair): 
					leftpair =  self.roullete_wheel()  # np.random.randint(self.pop_size) 
					rightpair = self.roullete_wheel()  # np.random.randint(self.pop_size) 

				first_child, second_child = self.xover_mutate(leftpair,rightpair) 
 
	   
				self.new_pop[i,:] = first_child 
				self.new_pop[i+1 ,:] = second_child



			best = self.pop[self.best_index, :]

			if self.best_fit > global_bestfit:
				global_bestfit = self.best_fit 
				global_best = self.pop[self.best_index, :]
 

			self.pop = self.new_pop


			self.pop[0,:] = global_best # ensure that you retain the best 

  
 
			if  (1/self.best_fit) < self.min_error:
				print(' reached min error')
				break 
  

		return best, 1/self.best_fit, global_best, 1/global_bestfit, self.pop, self.fit_list

 

def main():


	file=open('results_fitness.txt','a')

	file_solution=open('results_solution.txt','a')


	random.seed(time.time()) 

	min_fitness = 0.005  # stop when fitness reaches this value. not implemented - can be implemented later


	max_evals = 100000   # need to decide yourself - function evaluations 

	pop_size = 50  # to be adjusted for the problem
	num_variables =  5 # depends on your problem

	xover_rate = 0.8 # ideal, but you can adjust further 
	mu_rate = 0.1

 


	#max_limits = [2, 2, 2, 2, 2]  # can handle different limits for different variables
	max_limits = np.repeat(2, num_variables)
	#min_limits = [0, 0, 0, 0, 0] 

	min_limits = np.repeat(0, num_variables)

	prob = 1 # 1 is rosenbrock, 2 is ellipsoidal

	num_experiments = 5 # number of experiments with diffirent initial population
	num_xover = 3

	# lets evaluate which one is the best xover operator, 

	#pre results show that for 10 dimensions - blend xover is best, then wrights, then sim binary xover - for Rosenbrock problem
	#wrights heuristic works best for ellipsoidal problem

	global_fit = np.zeros(num_experiments)


	for i in range(num_xover):


		choose_xover  = i +1 # 1. Wrights Heuristic Xover, 2. Simulated Binary Xover, 3. Blend xover, 4.  

		for j in range(num_experiments): 

			evo = Evolution(prob, pop_size, num_variables, max_evals,  max_limits, min_limits, xover_rate, mu_rate, min_fitness, choose_xover)
 
			best, best_fit, global_best, global_bestfit, pop, fit_list = evo.evo_alg()
			np.savetxt(file_solution, np.transpose([i,j, global_bestfit, best_fit]), fmt='%1.1f')
			np.savetxt(file_solution, np.transpose(global_best) , fmt='%1.4f') 
			np.savetxt(file_solution, np.transpose(best) , fmt='%1.4f')

			global_fit[j] = global_bestfit 

			print(j, best_fit, global_bestfit, ' best and global best fit')

			print(j, best, global_best, ' best and global best sol')


		print(global_fit, '    global bestfit')  
		print(i, np.mean(global_fit), '  mean  global bestfit') 
		#np.savetxt(file, global_fit)
 
		np.savetxt(file, [i,np.mean(global_fit) ], fmt='%1.2f')
 



 

if __name__ == "__main__": main()
