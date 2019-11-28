import numpy as np
from world_generator import *
import saver


class Bee:

    def __init__(self, world, neighbor_func, fitness_func):
        self.world = world
        self.neighbor_function = neighbor_func
        self.fitness_function = fitness_func
        self.current_solution = None

    def generate_new_solution(self, print_all=False):
        # get random city in current_solution here, and let function generate solution?
        nbors_sols = self.neighbor_function(self.world, self.current_solution, 5)
        # get best solution out of list?
        fitnesses = [self.fitness_function(self.world, x) for x in nbors_sols]

        if print_all:
            print("Bee:")
            print("\tneighborhood: ", nbors_sols)
            print("\tbest fitness ", max(fitnesses), 'out of\t', fitnesses)

        fit_sol = list(zip(fitnesses, nbors_sols))
        fit_sol.sort(key=lambda x: x[0], reverse=True)
        self.current_solution = fit_sol[0][1]


class BeesAlgorithm:

    # n - number of bees
    # good_n - number of bees to assign to good solutions
    # elite_n - number of bees to assign to elite solutions (as part of good bees)
    # good_sols - number of good solutions out of all solutions
    # elite_sols - number of elite solutions out of GOOD solutions
    # neighbor_func - function that returns all neighbors of a city
    # fit_func - fitness function: takes world and solution and returns single value
    def __init__(self, world, n, good_n, elite_n, good_sols, elite_sols, neighbor_func, fit_func):
        self.world = world
        self.n_of_bees = n
        self.n_of_good = good_n
        self.n_of_elite = elite_n
        self.good_solutions = good_sols
        self.elite_solutions = elite_sols
        self.neighbor_function = neighbor_func
        self.fitness_function = fit_func

    def run(self, max_iter, solutions, print_all=False):
        bees = [Bee(self.world, self.neighbor_function, self.fitness_function) for x in range(0, self.n_of_bees)]
        best_solution = None
        best_solution_overall = None
        best_solution_fitness_overall = 0
        worst_solution_fitness_overall = 10000
        best_solution_iteration = 0
        iteration = 0

        best_fitnesses = []

        while iteration < max_iter:
            print("Iteration ", iteration + 1)
            solutions = self.iterate(bees, solutions, print_all)

            fitnesses = [self.fitness_function(self.world, x) for x in solutions]
            fit_sol = list(zip(fitnesses, solutions))
            fit_sol.sort(key=lambda x: x[0], reverse=True)

            best_solution = fit_sol[0][1]
            if fit_sol[0][0] > best_solution_fitness_overall:
                best_solution_overall = best_solution
                best_solution_iteration = iteration+1
                best_solution_fitness_overall = fit_sol[0][0]
                if fit_sol[-1][0] < worst_solution_fitness_overall:
                    worst_solution_fitness_overall = fit_sol[-1][0]
            print("Best fitness in iteration ", iteration + 1, " : ", fit_sol[0][0], " solution: ", best_solution)
            iteration += 1
            best_fitnesses.append(fit_sol[0][0])
            
        saver.save(self.world, 'world')
        saver.save(best_fitnesses, 'fitness_history')
        saver.save(best_solution, 'best_solution')

        print('Worst fitness overall: ', worst_solution_fitness_overall)
        print('Best solution: ', best_solution_overall)
        print('Found in ', best_solution_iteration, ' iteration')
        print('Fitness: ', best_solution_fitness_overall)

        return best_solution, best_fitnesses

    def iterate(self, bees, solutions, print_all=False):
        # calculate fitnesses
        fitnesses = [self.fitness_function(self.world, x) for x in solutions]
        fit_sol = list(zip(fitnesses, solutions))
        fit_sol.sort(key=lambda x: x[0], reverse=True)

        # recruit bees to solutions
        elite_sols = [s for (f, s) in fit_sol[:self.elite_solutions]]
        good_sols = [s for (f, s) in fit_sol[self.elite_solutions:self.good_solutions]]
        rest_sols = [s for (f, s) in fit_sol[self.good_solutions:]]
        for i, bee in enumerate(bees[:self.n_of_elite]):
            bee.current_solution = elite_sols[i % len(elite_sols)]
        for i, bee in enumerate(bees[self.n_of_elite:self.n_of_good]):
            bee.current_solution = good_sols[i % len(good_sols)]
        for i, bee in enumerate(bees[self.n_of_good:]):
            bee.current_solution = rest_sols[i % len(rest_sols)]

        # activate bees
        for bee in bees:
            bee.generate_new_solution(print_all)
        print('Fitnesses: ', [self.fitness_function(self.world, bee.current_solution) for bee in bees])
        # get new solutions
        return [bee.current_solution for bee in bees]


def neighborhood_function(world, solution, n_of_neighbors):
    def distinct(l):
        return list(set(l))

    neighbours = []

    for i in range(n_of_neighbors):
        # generate beginning and end of change
        beg = random.randrange(0, len(solution))
        end = beg
        while beg is end:
            end = random.randrange(0, len(solution))
        if end < beg:
            beg, end = end, beg

        # generate new road_matrix
        road_matrix = world.road_matrix.copy()
        cost_matrix = world.cost_matrix.copy()
        value_matrix = world.value_matrix.copy()
        size_of_the_world = np.size(road_matrix, 0)

        for j in range(0, beg):
            road_matrix[solution[j]] = np.zeros(size_of_the_world)
            road_matrix[:, solution[j]] = np.zeros(size_of_the_world)
            cost_matrix[solution[j]] = np.zeros(size_of_the_world)
            cost_matrix[:, solution[j]] = np.zeros(size_of_the_world)
            value_matrix[solution[j]] = np.zeros(size_of_the_world)
            value_matrix[:, solution[j]] = np.zeros(size_of_the_world)
        for j in range(end + 1, len(solution)):
            road_matrix[solution[j]] = np.zeros(size_of_the_world)
            road_matrix[:, solution[j]] = np.zeros(size_of_the_world)
            cost_matrix[solution[j]] = np.zeros(size_of_the_world)
            cost_matrix[:, solution[j]] = np.zeros(size_of_the_world)
            value_matrix[solution[j]] = np.zeros(size_of_the_world)
            value_matrix[:, solution[j]] = np.zeros(size_of_the_world)

        # generate change
        list_of_changes = generate_solutions(
            solution[beg],
            solution[end],
            road_matrix,
            cost_matrix,
            value_matrix,
            1
        )
        if len(list_of_changes) > 0:
            change = list_of_changes[0]

            # appending new solution to the list
            neighbours.append(solution[:beg] + change + solution[end + 1:])

    return distinct(neighbours)


def fitness_function(world, solution):
    value = 0
    for i in range(len(solution) - 1):
        value += world.value_matrix[solution[i]][solution[i + 1]]
        value -= world.cost_matrix[solution[i]][solution[i + 1]]
    return value


def main():
    # world parameters
    cities_no = 30
    world_size = 10

    # bee algorithm parameters
    iteration_no = 300
    no_of_solutions = 50
    good_n = 10
    elite_n = 5
    good_sols = 2
    elite_sols = 1

    w = World(cities_no, world_size)
    # w = saver.load('world')
    sols = generate_solutions(0, cities_no - 1, w.road_matrix, w.cost_matrix, w.value_matrix, no_of_solutions)
    fits = [fitness_function(w, sol) for sol in sols]
    sols = sols
    print("Generated ", len(sols), ' solutions:\n', sols)
    # print("Neighborhood function:\n", neighborhood_function(w, sols[0], 5))
    print('Best fitness: ', max(fits))

    alg = BeesAlgorithm(w, no_of_solutions, good_n, elite_n, good_sols, elite_sols, neighborhood_function, fitness_function)
    best_generated_solution, fitness_history = alg.run(iteration_no, sols)

    # from plotter import plot_history
    # plot_history(fitness_history)

    print("Best solution of all: ", best_generated_solution, " fitness: ", fitness_function(w, best_generated_solution))


if __name__ == '__main__':
    main()
