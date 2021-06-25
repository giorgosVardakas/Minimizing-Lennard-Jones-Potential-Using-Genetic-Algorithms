#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <omp.h>

double calculate_euclidean_distance(double *atom_i, double *atom_j)
{
	return sqrt(pow(atom_i[0] - atom_j[0], 2) + pow(atom_i[1] - atom_j[1], 2) + pow(atom_i[2] - atom_j[2], 2));
}

double calculate_dynamic_energy(double euclidean_distance, double sigma)
{
	double number;
	if (euclidean_distance != 0)
	{
		number = sigma / euclidean_distance;
	}
	else
	{
		number = DBL_MAX;
	}
	return pow(number, 12) - pow(number, 6);
}

double lennard_jones_function(double *atoms_position, int n, double epsilon, double sigma)
{
	double total_energy = 0.0;
	int i, j;

	for (i = 0; i < n - 1; ++i)
	{
		for (j = i + 1; j < n; ++j)
		{
			double euclidean_distance = calculate_euclidean_distance(atoms_position + i*3, atoms_position + j*3);
			double dynamic_energy = calculate_dynamic_energy(euclidean_distance, sigma);
			total_energy += dynamic_energy;
		}
	}

	total_energy *= 4 * epsilon;
	return total_energy;
}

void evaluate(double *population, double *values, int population_size, int number_of_atoms){
	int i;

	#pragma omp parallel for
	for(i = 0; i < population_size; ++i)
	{
		values[i] = lennard_jones_function(&population[i * number_of_atoms * 3], number_of_atoms, 1, 1);
	}
}
