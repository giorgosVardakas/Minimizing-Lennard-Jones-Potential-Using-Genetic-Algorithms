import numpy as np
import pandas as pd
from numpy.linalg.linalg import LinAlgError
from numpy.linalg import norm
import matplotlib.pyplot as plt
import sys
from ctypes import CDLL, POINTER
from ctypes import c_int, c_double

# Load the library I created for extra speed
mylib = CDLL("./mylib.so")

# C-type corresponding to numpy 2-dimensional array (matrix)
ND_POINTER_1 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C")
ND_POINTER_2 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C")
ND_POINTER_3 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=3, flags="C")

# define the prototypes of the functions
mylib.lennard_jones_function.argtypes = [ND_POINTER_2, c_int, c_double, c_double]
mylib.lennard_jones_function.restype = c_double

mylib.evaluate.argtypes = [ND_POINTER_3, ND_POINTER_2, c_int, c_int]
mylib.evaluate.restype = None

# For Genetic Algorithms
# Evaluation
def evaluate_population(population, number_of_atoms):
    values = np.zeros(shape=(population.shape[0], 1), dtype=np.float64)
    mylib.evaluate(population, values, population.shape[0], number_of_atoms)
    x_best_index = np.argmin(values)
    return values, x_best_index, values.min()

# Selection
def roulette_wheel_selection(population, evaluations, selective_pressure):
    descenting_order = np.argsort(evaluations, axis=0)[::-1]
    population = population[descenting_order]
    N = evaluations.shape[0]
    fitness_scores = np.zeros(shape=(N, 1))
    random_vector = np.random.uniform(low=0, high=1, size=(N, 1))
    selected_indexs = np.zeros(shape=(N, 1), dtype=int)

    for i, _ in enumerate(fitness_scores):
        fitness_scores[i] = 2 - selective_pressure + 2 * (selective_pressure - 1) * (i - 1) / (N - 1)

    selection_probabilities = fitness_scores / np.sum(fitness_scores)

    for rn_index, random_number in enumerate(random_vector):
        probability_sum = 0
        for sp_index, selection_probability in enumerate(selection_probabilities):
            probability_sum += selection_probability
            if random_number <= probability_sum:
                selected_indexs[rn_index] = sp_index
                break

    return np.squeeze(population[selected_indexs])

def tournament_selection(population, evaluations, tournament_size, dtype):
    N = population.shape[0]
    tournament_winners = np.zeros(shape=population.shape, dtype=dtype)

    for i in range(0, N):
        random_choices = np.random.choice(N, size=tournament_size, replace=False)
        tournament_winner_index = evaluations[random_choices].argmin()
        tournament_winners[i] = population[random_choices][tournament_winner_index]

    return tournament_winners

def new_population_top_N(population, mutated_population, population_evaluations, mutated_population_evaluations):
    N = population.shape[0]

    all_population = np.stack((population, mutated_population), axis=0)
    all_population = all_population.reshape((2 * population.shape[0], population.shape[1]))

    all_evaluations = np.stack((population_evaluations, mutated_population_evaluations))
    all_evaluations = all_evaluations.reshape((2 * population_evaluations.shape[0], 1))

    ascending_order = np.argsort(all_evaluations, axis=0)

    all_evaluations = all_evaluations[ascending_order]
    all_evaluations = all_evaluations.reshape((all_evaluations.shape[0], 1))

    all_population = all_population[ascending_order]
    all_population = np.squeeze(all_population)

    return all_population[0:N], all_evaluations[0:N]

# Genetic Algorithm Binary
def calculate_number_of_bits(Umin, Umax, error):
    length_of_space = Umax - Umin
    possible_numbers = 1 + length_of_space / error
    for n in range(1, 64):
        if np.power(2, n-1) < possible_numbers <= np.power(2, n):
            return n

def calculate_base_10(binary_number):
    number_base_10 = 0
    for i, bi in enumerate(binary_number):
        number_base_10 += bi * np.power(2, i)
    return number_base_10

def calculate_number_base_10_in_feasible_space(Umin, Umax, n_bits, number_base_10):
    length_of_space = Umax - Umin
    return Umin + number_base_10 * length_of_space / (np.power(2, n_bits) - 1)

def decoder(population, Umin, Umax, number_of_atoms, dimensionality, n_bits):
    population_base_10 = np.zeros(shape=(population.shape[0], number_of_atoms, dimensionality))

    for i, pi in enumerate(population):
        pi = np.array_split(pi, number_of_atoms)

        for j, pij in enumerate(pi):
            pij = np.array_split(pij, dimensionality)
            pij_base_10 = list()

            for binary_number in pij:
                number_base_10 = calculate_base_10(binary_number)
                number_base_10_fs = calculate_number_base_10_in_feasible_space(Umin, Umax, n_bits, number_base_10)
                pij_base_10.append(number_base_10_fs)

            population_base_10[i][j] = np.asarray(pij_base_10)

    return population_base_10

def initialize_binary_population(population_size, number_of_atoms, dimensionality, n_bits):
    population = np.random.randint(low=0, high=2, size=(population_size, number_of_atoms, dimensionality * n_bits))
    population = population.reshape(population_size, number_of_atoms * dimensionality * n_bits)
    return population

def crossover_binary_population(selected_population, crossover_rate, crossover_points):
    # crossover_rate = [0, 1]
    # crossover_points = m - 1, where m is the length of the dna

    N = selected_population.shape[0]
    to_crossover = np.random.uniform(low=0, high=1, size=(N, 1)) < crossover_rate
    to_crossover_indexes = np.where(np.any(to_crossover==True, axis=1))[0]
    crossover_population = np.array(selected_population)

    if to_crossover_indexes.shape[0] % 2 != 0:
        random_choice = np.random.randint(low=0, high=N)
        to_crossover_indexes = np.append(to_crossover_indexes, random_choice)

    parents = selected_population[to_crossover_indexes]
    children = np.zeros(shape=(parents.shape[0], parents.shape[1]), dtype=int)

    if parents.shape[0] == 0: return selected_population

    points_of_crossover = np.arange(1, selected_population.shape[1])
    np.random.shuffle(points_of_crossover)
    points_of_crossover = points_of_crossover[:crossover_points]
    points_of_crossover = np.sort(points_of_crossover, axis=0)

    for i in range(0, parents.shape[0], 2):
        parent_0 = np.array_split(parents[i], points_of_crossover)
        parent_1 = np.array_split(parents[i + 1], points_of_crossover)
        child_0, child_1 = list(), list()

        for j in range(0, crossover_points + 1):
            if j % 2 == 0:
                child_0.append(parent_0[j])
                child_1.append(parent_1[j])
            else:
                child_0.append(parent_1[j])
                child_1.append(parent_0[j])

        child_0 = np.asarray(child_0, dtype=object)
        child_1 = np.asarray(child_1, dtype=object)
        children[i] = np.concatenate(child_0, axis=None)
        children[i + 1] = np.concatenate(child_1, axis=None)

    # Replace parents with their children
    for child_index, parent_index in enumerate(to_crossover_indexes):
        crossover_population[parent_index] = children[child_index]

    return crossover_population

def mutation_binary_population(crossover_population, mutation_rate):
    # mutation_rate = [0, 1]
    mutated_population = np.array(crossover_population)
    for i, pi in enumerate(mutated_population):
        to_mutate = np.random.uniform(low=0, high=1, size=(pi.shape[0], 1)) < mutation_rate
        to_mutate_indexes = np.where(np.any(to_mutate==True, axis=1))[0]
        for j in to_mutate_indexes:
            pi[j] = 1 - pi[j]
        mutated_population[i] = pi
    return mutated_population

def Genetic_Algorithm_Binary(Umin, Umax, number_of_atoms, selection_method):
    # Algorithm parameters
    population_size = 1000
    selective_pressure = 1.3 # selective_pressure = [1, 2]
    tournament_size = 100 # tournament_size = [1, population_size]
    crossover_rate = 0.5 # crossover_rate = [0, 1]
    crossover_points = 6 # crossover_points = [0, m-1]
    mutation_rate = 0.1 # mutation_rate = [0, 1]

    # Do not change
    error = 1e-3
    dimensionality = 3 # 3D space
    n_bits = calculate_number_of_bits(Umin, Umax, error)
    iteration = population_size
    best_iteration = iteration
    max_iterations = number_of_atoms * 1e+5

    population = initialize_binary_population(population_size, number_of_atoms, dimensionality, n_bits)
    decoded_population = decoder(population, Umin, Umax, number_of_atoms, dimensionality, n_bits)
    population_evaluations, x_best_index, min_value = evaluate_population(decoded_population, number_of_atoms)

    x_best = population[x_best_index]
    x_best_fvalue = min_value

    while(True):
        if (max_iterations <= iteration): break
        iteration += population_size

        if(selection_method == "rw"):
            selected_population = roulette_wheel_selection(population, population_evaluations, selective_pressure)
        else:
            selected_population = tournament_selection(population, population_evaluations, tournament_size, int)
        crossover_population = crossover_binary_population(selected_population, crossover_rate, crossover_points)
        mutated_population = mutation_binary_population(crossover_population, mutation_rate)
        decoded_population = decoder(mutated_population, Umin, Umax, number_of_atoms, dimensionality, n_bits)
        mutated_population_evaluations, _, _ = evaluate_population(decoded_population, number_of_atoms)
        population, population_evaluations = new_population_top_N(population, mutated_population, population_evaluations, mutated_population_evaluations)

        if population_evaluations[0] < x_best_fvalue:
            x_best, x_best_fvalue = population[0], population_evaluations[0]
            best_iteration = iteration
            print("Iterations: %d/%d Lennard-Jones potential: %.10f!!!" % (iteration, max_iterations, x_best_fvalue))
        else:
            print("Iterations: %d/%d Lennard-Jones potential: %.10f" % (iteration, max_iterations, x_best_fvalue))

    x_best = x_best.reshape((1, x_best.size))
    population = np.append(population, x_best, axis=0)
    decoded_population = decoder(population, Umin, Umax, number_of_atoms, dimensionality, n_bits)
    population_evaluations, x_best_index, x_best_fvalue = evaluate_population(decoded_population, number_of_atoms)
    x_best = decoded_population[x_best_index]

    return x_best, x_best_fvalue, decoded_population, population_evaluations, best_iteration

# Genetic Algorithm Real
def initialize_real_population(Umin, Umax, population_size, number_of_atoms, dimensionality):
    population = np.random.uniform(low=Umin, high=Umax, size=(population_size, number_of_atoms, dimensionality))
    return population

def crossover_real_population(selected_population, crossover_rate, delta=0.25):
    # crossover_rate = [0, 1]
    # delta > 0
    N = selected_population.shape[0]
    to_crossover = np.random.uniform(low=0, high=1, size=(N, 1)) < crossover_rate
    to_crossover_indexes = np.where(np.any(to_crossover==True, axis=1))[0]
    crossover_population = np.array(selected_population)

    if to_crossover_indexes.shape[0] % 2 != 0:
        random_choice = np.random.randint(low=0, high=N)
        to_crossover_indexes = np.append(to_crossover_indexes, random_choice)

    parents = selected_population[to_crossover_indexes]
    children = np.zeros(shape=parents.shape, dtype=float)

    if parents.shape[0] == 0: return selected_population

    for i in range(0, parents.shape[0], 2):
        # Create a pair of children for a pair of parents
        for j in range(0, 2):
            random_vector = np.random.uniform(low=-delta, high=1+delta, size=selected_population.shape[1])
            child = np.multiply(random_vector, parents[i]) + np.multiply((1 - random_vector), parents[i + 1])
            children[i + j] = child

    # Replace parents with their children
    for child_index, parent_index in enumerate(to_crossover_indexes):
        crossover_population[parent_index] = children[child_index]

    return crossover_population

def mutation_real_population(crossover_population, mutation_rate, Umin, Umax):
    # mutation_rate = [0, 1]
    mutated_population = np.array(crossover_population)
    for i, pi in enumerate(mutated_population):
        to_mutate = np.random.uniform(low=0, high=1, size=(pi.shape[0], 1)) < mutation_rate
        to_mutate_indexes = np.where(np.any(to_mutate==True, axis=1))[0]
        for j in to_mutate_indexes:
            distance_from_Umin = abs(abs(pi[j]) - abs(Umin))
            distance_from_Umax = abs(abs(pi[j]) - abs(Umax))
            min_distance = min(distance_from_Umin, distance_from_Umax)
            sigma = min_distance / 3
            zj = np.random.normal(0, sigma)
            pi[j] = pi[j] + zj
        mutated_population[i] = pi

    return mutated_population

def Genetic_Algorithm_Real(Umin, Umax, number_of_atoms, selection_method):
    # Algorithm parameters
    population_size = 1000
    selective_pressure = 1.3
    tournament_size = 100
    crossover_rate = 0.5
    mutation_rate = 0.1

    # Do not change
    dimensionality = 3
    iteration = population_size
    best_iteration = iteration
    max_iterations = number_of_atoms * 1e+5

    population = initialize_real_population(Umin, Umax, population_size, number_of_atoms, dimensionality)
    population_evaluations, x_best_index, min_value = evaluate_population(population, number_of_atoms)

    # vectorize population for the alogrithm
    population = population.reshape(population_size, number_of_atoms * dimensionality)

    x_best = population[x_best_index]
    x_best_fvalue = min_value

    while(True):
        if (max_iterations <= iteration): break
        iteration += population_size

        if selection_method == "rw":
            selected_population = roulette_wheel_selection(population, population_evaluations, selective_pressure)
        else:
            selected_population = tournament_selection(population, population_evaluations, tournament_size, float)
        crossover_population = crossover_real_population(selected_population, crossover_rate)
        mutated_population = mutation_real_population(crossover_population, mutation_rate, Umin, Umax)

        # create mutated_population as array for evaluation
        array_mutated_population = mutated_population.reshape(mutated_population.shape[0], number_of_atoms, dimensionality)
        mutated_population_evaluations, _, _ = evaluate_population(array_mutated_population, number_of_atoms)
        population, population_evaluations = new_population_top_N(population, mutated_population, population_evaluations, mutated_population_evaluations)

        if population_evaluations[0] < x_best_fvalue:
            x_best, x_best_fvalue = population[0], population_evaluations[0]
            best_iteration = iteration
            print("Iterations: %d/%d Lennard-Jones potential: %.10f!!!" % (iteration, max_iterations, x_best_fvalue))
        else:
            print("Iterations: %d/%d Lennard-Jones potential: %.10f" % (iteration, max_iterations, x_best_fvalue))

    x_best = x_best.reshape(number_of_atoms, dimensionality)
    population = population.reshape(population.shape[0], number_of_atoms, dimensionality)

    return x_best, x_best_fvalue.item(), population, population_evaluations, best_iteration

# Particle Swarm Optimization
def initialize_velocity(swarm_size, number_of_atoms, dimensionality, max_velocity):
    velocity = np.random.uniform(low=-max_velocity, high=max_velocity, size=(swarm_size, number_of_atoms, dimensionality))
    return velocity

def create_neighborhoods(swarm_size, neighborhood_radius):
    # neighborhood_radius = [0, N/2]
    neighborhoods = list()
    for i in range(0, swarm_size):
        neighborhood_i = list()
        for j in range(i - neighborhood_radius, i + neighborhood_radius + 1):
            neighborhood_i.append(j % swarm_size)
        neighborhoods.append(np.asarray(neighborhood_i))
    return np.asarray(neighborhoods)

def update_velocity(swarm, velocity, best_positions, best_positions_evaluations, neighborhoods, c1=2.05, c2=2.05, x=0.729):
    rgn_0 = np.random.uniform(low=0, high=1)
    rgn_1 = np.random.uniform(low=0, high=1)
    best_neighbors = np.zeros(shape=swarm.shape)

    for particle_i, neighbors_i in enumerate(neighborhoods):
        best_neighbor_index = 0
        best_neighbor_f_value = best_positions_evaluations[neighbors_i[0]]
        for j, neighbor_ij in enumerate(neighbors_i):
            neighborij_f_value = best_positions_evaluations[neighbor_ij]
            if neighborij_f_value < best_neighbor_f_value:
                best_neighbor_index = j
                best_neighbor_f_value = neighborij_f_value
        best_neighbors[particle_i] = swarm[neighbors_i[best_neighbor_index]]
    velocity = x * (velocity + rgn_0 * c1 * (best_positions - swarm) + rgn_1 * c2 * (best_neighbors - swarm))
    return velocity

def update_particles(swarm, velocity):
    return swarm + velocity

def check_velocity_bounds(velocity, max_velocity):
    for i, vi in enumerate(velocity):
        for j, vij in enumerate(vi):
            if vij < -max_velocity:
                vij = -max_velocity
            elif max_velocity < vij:
                vij = max_velocity
            vi[j] =  vij
        velocity[i] = vi
    return velocity

def check_particles_bounds(swarm, Umin, Umax):
    for i, pi in enumerate(swarm):
        for j, pij in enumerate(pi):
            if pij < Umin:
                pij = Umin
            elif Umax < pij:
                pij = Umax
            pi[j] = pij
        swarm[i] = pi
    return swarm

def update_best_positions(best_positions, best_positions_evaluations, swarm, swarm_evaluations):
    for i, _ in enumerate(swarm_evaluations):
        if swarm_evaluations[i] < best_positions_evaluations[i]:
            best_positions[i] = swarm[i]
            best_positions_evaluations[i] = swarm_evaluations[i]
    return best_positions, best_positions_evaluations

def Particle_Swarm_Optimization(Umin, Umax, number_of_atoms, model):
    # Algorithm parameters
    swarm_size = 1000
    alpha = 0.5 # alpha = [0, 1]
    # neighborhood_radius = [0, swarm_size / 2]
    if model == "lbest":
        neighborhood_radius = 5
    else:
        neighborhood_radius = int(swarm_size / 2)

    # Do not change
    max_velocity = alpha * (Umax - Umin)
    iteration = swarm_size
    best_iteration = iteration
    max_iterations = number_of_atoms * 1e+5
    dimensionality = 3

    # Initializations
    swarm = initialize_real_population(Umin, Umax, swarm_size, number_of_atoms, dimensionality)
    velocity = initialize_velocity(swarm_size, number_of_atoms, dimensionality, max_velocity)
    best_positions = np.array(swarm)
    neighborhoods = create_neighborhoods(swarm_size, neighborhood_radius)
    best_positions_evaluations, x_best_index, min_value = evaluate_population(swarm, number_of_atoms)

    # Vectorization of the arrays for the alogrithm
    swarm = swarm.reshape(swarm_size, number_of_atoms * dimensionality)
    velocity = velocity.reshape(swarm_size, number_of_atoms * dimensionality)
    best_positions = best_positions.reshape(swarm_size, number_of_atoms * dimensionality)

    # Remember the best
    x_best = best_positions[x_best_index]
    x_best_fvalue = min_value

    while(True):
        if (max_iterations <= iteration): break
        iteration += swarm_size

        velocity = update_velocity(swarm, velocity, best_positions, best_positions_evaluations, neighborhoods)
        velocity = check_velocity_bounds(velocity, max_velocity)

        swarm = update_particles(swarm, velocity)
        swarm = check_particles_bounds(swarm, Umin, Umax)

        # Array-like swarm for evaluation
        array_swarm = swarm.reshape(swarm_size, number_of_atoms, dimensionality)
        swarm_evaluations, x_best_index, min_value = evaluate_population(array_swarm, number_of_atoms)
        best_positions, best_positions_evaluations = update_best_positions(best_positions, best_positions_evaluations, swarm, swarm_evaluations)
        best_index = best_positions_evaluations.argmin()

        if best_positions_evaluations[best_index] < x_best_fvalue:
            x_best = best_positions[best_index]
            x_best_fvalue = best_positions_evaluations[best_index]
            best_iteration = iteration
            print("Iterations: %d/%d Lennard-Jones potential: %.10f!!!" % (iteration, max_iterations, x_best_fvalue))
        else:
            print("Iterations: %d/%d Lennard-Jones potential: %.10f" % (iteration, max_iterations, x_best_fvalue))

    x_best = x_best.reshape(number_of_atoms, dimensionality)
    best_positions = best_positions.reshape(swarm_size, number_of_atoms, dimensionality)

    return x_best, x_best_fvalue.item(), best_positions, best_positions_evaluations, best_iteration

def main():
    if(len(sys.argv) != 2):
        print("Error: Wrong input.")
        print("Usage: python3 optimization.py number_of_atoms")
        print("Example: python3 optimization.py 4")
        exit()

    seeds = pd.read_csv("seeds.csv", header=None)
    seeds = seeds.to_numpy()

    Umin, Umax = -2.5, 2.5
    number_of_atoms = int(sys.argv[1])

    df_results = pd.DataFrame()
    results = dict()

    for seed_index, seed in enumerate(seeds):
        print("Number of atoms: %d, Expiriment: %d" % (number_of_atoms, seed_index))
        np.random.seed(seed)

        print("Real Genetic Algorithm with tournament selection.")
        _, GA_Real_t_best, _, _, GA_Real_t_iteration = Genetic_Algorithm_Real(Umin, Umax, number_of_atoms, selection_method="t")

        print("Particle Swarm Optimization with local best.")
        _, PSO_lbest_best, _, _, PSO_lbest_iteration = Particle_Swarm_Optimization(Umin, Umax, number_of_atoms, model="lbest")

        print("Particle Swarm Optimization with global best.")
        _, PSO_gbest_best, _, _, PSO_gbest_iteration = Particle_Swarm_Optimization(Umin, Umax, number_of_atoms, model="gbest")

        print("Binary Genetic Algorithm with roulette wheel selection.")
        _, GA_Binary_rw_best, _, _, GA_Binary_rw_iteration = Genetic_Algorithm_Binary(Umin, Umax, number_of_atoms, selection_method="rw")

        print("Binary Genetic Algorithm with tournament selection.")
        _, GA_Binary_t_best, _, _, GA_Binary_t_iteration = Genetic_Algorithm_Binary(Umin, Umax, number_of_atoms, selection_method="t")

        results["GA_Binary_rw"] = GA_Binary_rw_best
        results["GA_Binary_t"] = GA_Binary_t_best
        results["GA_Real_t"] = GA_Real_t_best
        results["PSO_lbest"] = PSO_lbest_best
        results["PSO_gbest"] = PSO_gbest_best

        results["GA_Binary_rw_iteration"] = GA_Binary_rw_iteration
        results["GA_Binary_t_iteration"] = GA_Binary_t_iteration
        results["GA_Real_t_iteration"] = GA_Real_t_iteration
        results["PSO_lbest_iteration"] = PSO_lbest_iteration
        results["PSO_gbest_iteration"] = PSO_gbest_iteration

        df_data = pd.Series(data=results)
        df_results = df_results.append(df_data, ignore_index=True)
        df_results.to_excel("./Results/Results_N" + str(number_of_atoms) + ".xlsx")


if __name__ == '__main__':
    main()
