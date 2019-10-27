import world_generator

class Bee:
	
	def __init__(self, world, neighbor_func):
		self.world = world
		self.neighbor_function = neighbor_func
		self.current_solution = None

	def generate_new_solution(self):
		pass #TODO

class BeesAlgorithm:

	# n - number of bees
	# good_n - number of bees to assign to good solutions
	# elite_n - number of bees to assign to elite solutions (as part of good bees)
	# good_sols - number of good solutions out of all solutions
	# elite_sols - number of elite solutions out of GOOD solutions
	# neighbor_func - function that returns all neighbors of a city
	# fit_func - fitness function: takes world and solution and returns single value
	def __init__(self, world, n, good_n, elite_n, 
				good_sols, elite_sols, neighbor_func, fit_func):
		self.world = world
		self.n_of_bees = n
		self.n_of_good = good_n
		self.n_of_elite = elite_n
		self.good_solutions = good_sols
		self.elite_solutions = elite_sols
		self.neighbor_function = neighbor_func
		self.fitness_function = fit_func

	def run(self, max_iter, solutions):
		bees = [Bee(self.world) for x in range(0, self.n_of_bees)]	
		best_solution = None
		iteration = 0
		while iteration < max_iter:
			solutions = iterate(bees, solutions)
			
			fitnesses = [self.fitness_function(self.world,x) for x in solutions]
			fit_sol = list(zip(fitnesses, solutions))
			fit_sol.sort(key = lambda x: x[0])
			
			best_solution = fit_sol[0][1]
			iteration += 1
		return best_solution

	def iterate(self, bees, solutions):
		#calculate fitnesses
		fitnesses = [self.fitness_function(self.world,x) for x in solutions]
		fit_sol = list(zip(fitnesses, solutions))
		fit_sol.sort(key = lambda x: x[0])
		#recruit bees to solutions
		elite_sols = [s for (f,s) in fit_sol[:self.elite_solutions]]
		good_sols = [s for (f,s) in fit_sol[self.elite_solutions:self.good_solutions]]
		rest_sols = [s for (f,s) in fit_sol[self.good_solutions:]]
		for i, bee in enumerate(bees[:self.n_of_elite]):
			bee.current_solution = elite_sols[i % len(elite_sols)]
		for i, bee in enumerate(bees[self.n_of_elite:self.n_of_good]):
			bee.current_solution = good_sols[i % len(good_sols)]
		for i, bee in enumerate(bees[self.n_of_good:]):
			bee.current_solution = rest_sols[i % len(rest_sols)]
		#activate bees
		for bee in bees:
			bee.generate_new_solution()
		#get new solutions
		return [x for bee.current_solution in bees]

	

def main():
	cities_no = 10
    world_size = 10

    w = World(cities_no, world_size)
    no_of_solutions = 30
    sols = generate_solutions(0, cities_no - 1, w.road_matrix, w.cost_matrix, w.value_matrix, no_of_solutions)
    print("Solutions:\n", len(sols))
    print(sols)
	# alg = BeesAlgorithm(w, 35, 15, 5, 15, 5, ???, ???)

if __name__ == '__main__':
	main()