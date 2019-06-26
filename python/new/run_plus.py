# General Imports
import sys
import argparse
from tqdm import tqdm

# Math Imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.sparse.linalg as la
import scipy.interpolate as interpolate
import random

# Shared functions
import shared
import plus

def setup(b, l, crosswires, level):
    # Begin Setup for Calculating Harmonic Function
    print('Beginning Setup for + Graph Approximation using b=%d, l=%d, crosswires=%d, level=%d ...' % (b, l, crosswires, level))
    grid_size = plus.get_grid_size(b, crosswires, level)
    layout = plus.get_grid_layout(b, l, crosswires, level)

    # Visualization of Fractal
    shared.display_grid_layout(layout, display_type='matplotlib')

    # Possibly need to clear some memory, insert `del layout` at some point
    coordinates = shared.index_layout(layout)
    adjacency_list = plus.get_adjacency_list(layout, coordinates, crosswires)
    laplacian = shared.compute_laplacian(adjacency_list)

    return grid_size, layout, coordinates, adjacency_list, laplacian

def left_to_right_potentials(b, crosswires, level):
    edge_length = crosswires*b**level

    # Calculate Boundary Indices
    boundary_indices = []
    boundary_indices.extend(range(edge_length))
    boundary_indices.extend(range(2*edge_length, 3*edge_length))

    # Set Dirichlet Boundary
    boundary = np.zeros((2*edge_length))
    boundary[edge_length:] = 1

    potentials = shared.compute_harmonic_function(laplacian, boundary_indices, boundary)
    harmonic_function = shared.display_harmonic_function(potentials, coordinates, grid_size, display_type='grid')

    return potentials, harmonic_function, boundary_indices

def left_to_bottom_potentials(b, crosswires, level):
    edge_length = crosswires*b**level

    # Calculate Boundary Indices
    boundary_indices = []
    boundary_indices.extend(range(2*edge_length))

    # Set Dirichlet Boundary
    boundary = np.zeros((2*edge_length))
    boundary[edge_length:] = 1

    potentials = shared.compute_harmonic_function(laplacian, boundary_indices, boundary)
    harmonic_function = shared.display_harmonic_function(potentials, coordinates, grid_size, display_type='grid')

    return potentials, harmonic_function, boundary_indices

def exit_distribution_potentials(b, crosswires, level):
    edge_length = crosswires*b**level

    # Set Dirichlet Boundary Indices
    boundary_indices = []
    boundary_indices.extend(range(4*edge_length))
    #print(plus.get_grid_size(b, crosswires, level-1)//2)
    boundary_indices.append(len(coordinates)- crosswires*b**(level-1)//2-1)

    # Set Dirichlet Boundary
    boundary = np.full((4*edge_length+1), 0)
    boundary[-1] = 1

    potentials = shared.compute_harmonic_function(laplacian, boundary_indices, boundary)
    harmonic_function = shared.display_harmonic_function(potentials, coordinates, grid_size, display_type='grid')

    return potentials, harmonic_function, boundary_indices

def potential_diff_distribution(adjacency_list, potentials, coordinates, grid_size):
    potentials_diff = np.empty((len(adjacency_list)))

    for i, row in enumerate(adjacency_list):
        max_diff = 0
        for index in row:
            diff = abs(potentials[i] - potentials[index])
            if diff > max_diff:
                max_diff = diff
        potentials_diff[i] = max_diff

    diff_display = np.full((grid_size, grid_size), None, dtype=float)

    # Generates table with harmonic function
    for i,coordinate in enumerate(coordinates):
        y, x = coordinate
        diff_display[y, x] = potentials_diff[i]

    plt.imshow(diff_display)
    plt.colorbar()
    plt.show()

def normal_derivative_boundary(adjacency_list, boundary_indices, potentials):
    ys = []
    for index in boundary_indices:
        row = adjacency_list[index]
        total = 0
        for neighbors in row:
            total += (potentials[neighbors]-potentials[index])
        ys.append(total)

    plt.plot(ys)
    plt.show()

    # Code for Chris
    a = ys[len(ys)/4:2*len(ys)/4]
    b = ys[3*len(ys)/4:]
    c = []
    for i, val in enumerate(a):
        c.append(b[i]/a[i])

    plt.plot(c)
    plt.show()


if __name__ == '__main__':
    # Make printing a bit nicer for visualizing
    np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

    # Algorithm Parameters (Type -h for usage)
    parser = argparse.ArgumentParser(description='Generates the + Graph Approximations for the Sierpinski Carpet')
    parser.add_argument('-b', default=3, type=int, help='The number of sections to divide the carpet into')
    parser.add_argument('-l', default=1, type=int, help='The number of sections to remove from the carpet center')
    parser.add_argument('-c', '--crosswires', type=int, default=1, help='The number of crosswires')
    parser.add_argument('-a', '--level', type=int, default=3, help='Number of pre-carpet contraction iterations')
    args = parser.parse_args()

    # General Setup script to make laplacian
    grid_size, layout, coordinates, adjacency_list, laplacian = setup(args.b, args.l, args.crosswires, args.level)

    # Left to Right (0 -> 1) Harmonic Function
    #potentials, harmonic_function, boundary_indices = left_to_right_potentials(args.b, args.crosswires, args.level)
    #potential_diff_distribution(adjacency_list, potentials, coordinates, grid_size)

    # Resistance Calculation

    #

    # Left to Bottom (0 -> 1) Harmonic Function
    #potentials2, harmonic_function2, boundary_indices2 = left_to_bottom_potentials(args.b, args.crosswires, args.level)

    # Exit Distribution Harmonic Function
    potentials3, harmonic_function3, boundary_indices3 = exit_distribution_potentials(args.b, args.crosswires, args.level)
    normal_derivative_boundary(adjacency_list, boundary_indices3[:-1], potentials3)

    # Resistance Calculation
    print('resistance', 1/shared.get_energy(adjacency_list, potentials3))

    # Max Edges
    max_edges = shared.max_edges(adjacency_list, potentials3, coordinates, grid_size)
    for edge in max_edges:
        print('------')
        print('left edge', coordinates[edge[0], 0], coordinates[edge[0], 1])
        print('right edge', coordinates[edge[1], 0], coordinates[edge[1], 1])
        print('left potential', potentials3[edge[0]])
        print('right potential', potentials3[edge[1]])
        print('------')





    # Max Edge Portion
    '''max_edges = shared.max_edges(adjacency_list, potentials, coordinates, grid_size)
    print(max_edges)
    min_coordinate = (-1, -1)
    for edge in max_edges:
        print('------')
        print('left edge', coordinates[edge[0], 0], coordinates[edge[0], 1])
        print('right edge', coordinates[edge[1], 0], coordinates[edge[1], 1])
        print('left potential', potentials[edge[0]])
        print('right potential', potentials[edge[1]])
        print('------')'''
