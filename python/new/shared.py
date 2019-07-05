# General Imports
from __future__ import print_function
import sys
import argparse
import datetime
from tqdm import tqdm

# Math Imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.sparse.linalg as la # (likely not needed)

# Shared functions
import shared

def display_grid_layout(layout, display_type='grid'):
    ''' For viewing the fractal (SC) layout. Two options are available
    ("terminal" and "matplotlib") '''

    if display_type == 'matplotlib':
        plt.imshow(layout.astype(int))
        plt.show()
    elif display_type == 'terminal':
        for y, row in enumerate(layout):
            for x, val in enumerate(row):
                if val == -2:
                    print(u"\u2588 ", end='')
                else:
                    print("  ", end='')
            print()


def display_harmonic_function(potentials, coordinates, grid_size, display_type='grid', num_contours=100, scale="linear"):
    ''' For viewing the harmonic functions computed. Two options are available
    ("grid" and "contour").  The harmonic function np array is returned (for
    convenience) '''

    harmonic_function = np.full((grid_size, grid_size), None, dtype=float)

    # Generates table with harmonic function
    for i,coordinate in enumerate(coordinates):
        y, x = coordinate
        harmonic_function[y, x] = potentials[i]

    # Selection of display option
    if display_type == 'grid':
        plt.imshow(harmonic_function, vmin=0, vmax=1)
        plt.colorbar()
        plt.show()

    elif display_type == 'contour':
        ys = range(grid_size)
        xs = range(grid_size)

        # Squares plot
        ax = plt.gca()
        ax.set_aspect('equal')

        if scale == "linear":
            plt.contour(ys, xs, harmonic_function, levels=np.linspace(0, 1, num=num_contours))
        elif scale == "cubic":
            plt.contour(ys, xs, harmonic_function, levels=np.linspace(0, 1, num=num_contours)**3)
        plt.colorbar()
        plt.show()

    return harmonic_function


def index_layout(layout):
    ''' Enumerates the layout of the given fractal.  At each location a -2 is
    replaced by the index of the coordinate. All vertices are indexed in
    counterclockwise order (spiraling inward) starting with the left edge. A
    list of coordinates are also returned'''

    print('Indexing the given layout ...')

    coordinates = []
    grid_size = np.shape(layout)[0]
    current_index = 0

    # This loop may need to be modified (ensuring progress bar but simpler)
    for removed_layers in tqdm(range(0, int(np.ceil(grid_size/2.0))), total=int(np.ceil(grid_size/2.0))):
        # Enumerates Left Edge
        for y in range(removed_layers, grid_size-removed_layers):
            if layout[y, removed_layers] == -2:
                layout[y, removed_layers] = current_index
                coordinates.append((y, removed_layers))
                current_index += 1

        # Enumerates Bottom Edge
        for x in range(removed_layers, grid_size-removed_layers):
            if layout[grid_size-1-removed_layers, x] == -2:
                layout[grid_size-1-removed_layers, x] = current_index
                coordinates.append((grid_size-1-removed_layers, x))
                current_index += 1

        # Enumerates Right Edge
        for y in range(grid_size-1-removed_layers, removed_layers-1,-1):
            if layout[y, grid_size-1-removed_layers] == -2:
                layout[y, grid_size-1-removed_layers] = current_index
                coordinates.append((y, grid_size-1-removed_layers))
                current_index += 1

        # Enumerates Top Edge
        for x in range(grid_size-1-removed_layers, removed_layers-1, -1):
            if layout[removed_layers, x] == -2:
                layout[removed_layers, x] = current_index
                coordinates.append((removed_layers, x))
                current_index += 1

        removed_layers +=1

    return np.array(coordinates)


def compute_laplacian(adjacency_list):
    ''' Computes a sparse matrix laplacian from the given adjacency list.
    Uses a lil_matrix currently'''

    print('Computing Laplacian ...')
    laplacian = sparse.lil_matrix((len(adjacency_list), len(adjacency_list)), dtype=np.short)

    # Creates a sparse matrix of the laplacian from the adjacency_list
    for i, row in tqdm(enumerate(adjacency_list), total=len(adjacency_list)):
        for index in row:
            laplacian[i, index] = 1

        laplacian[i, i] = -len(row)
        #print(-len(row))

    return sparse.csr_matrix(laplacian)


def compute_harmonic_function(laplacian, boundary_indices, boundary):
    print('Computing Harmonic Function Potentials ...')
    num_boundary_points = len(boundary_indices)
    num_computed_points = laplacian.get_shape()[0]-num_boundary_points
    total_points = num_boundary_points + num_computed_points

    # Calculating Permutation Matrix
    perm = sparse.lil_matrix((total_points, total_points), dtype=np.short)

    boundary_pos = 0
    computed_pos = 0

    for i in tqdm(range(total_points), total=total_points):
        if boundary_pos < num_boundary_points and i == boundary_indices[boundary_pos]:
            perm[boundary_pos, i] = 1
            boundary_pos+=1
        else:
            perm[num_boundary_points+computed_pos, i] = 1
            computed_pos+=1

    # Applying Permutation Matrix to put Boundary points first
    perm = sparse.csr_matrix(perm)
    reordered_laplacian = perm.dot(laplacian.dot(perm.transpose()))

    # Reorganize Matrices for Solver
    a = reordered_laplacian[num_boundary_points:, num_boundary_points:]
    r = reordered_laplacian[num_boundary_points:, :num_boundary_points]
    b = -r.dot(boundary)

    # Uses a linear algebra solver to compute harmonic function potentials
    potentials = la.spsolve(a, b)

    # Reinsert Boundary Potentials into solved potentials
    for i, boundary_index in enumerate(boundary_indices):
        potentials = np.insert(potentials, boundary_index, boundary[i])

    return potentials


def max_edges(adjacency_list, potentials, coordinates, grid_size, boundary_indices):
    ''' Finds the maximum edges in a given fractal (based on the associated
    adjacency matrix).  Returns the coordinates of these edges in a list
    (Should probably just return the indices based on coordinates****)'''

    max_diff = 0
    max_edges = []

    for i, row in enumerate(adjacency_list):
        for index in row:
            if i < index and i not in boundary_indices and index not in boundary_indices:
                min_potential_index = -1
                max_potential_index = -1

                # To ensure first element in edge is lower potential
                if potentials[index] < potentials[i]:
                    min_potential_index = index
                    max_potential_index = i
                else:
                    min_potential_index = i
                    max_potential_index = index

                diff = abs(potentials[i]-potentials[index])

                if abs(diff-max_diff) <= 0.0000001:
                    max_edges.append((min_potential_index, max_potential_index))
                elif diff - max_diff > 0.0000001:
                    max_diff = diff
                    max_edges = [(min_potential_index, max_potential_index)]


    return np.array(max_edges)

def max_chains(adjacency_list, potentials, coordinates, max_length=5):
    max_chains = np.zeros((max_length+1))

    for i, row in tqdm(enumerate(adjacency_list), total=len(adjacency_list)):
        stack = []
        starting_potential = potentials[i]
        stack.append((i, 0))

        while len(stack) > 0:
            index, chain_length = stack.pop(0)
            if chain_length <= max_length:
                if max_chains[chain_length] < potentials[index]-starting_potential:
                    max_chains[chain_length] = potentials[index]-starting_potential
            else:
                break

            for j in adjacency_list[index]:
                if potentials[j] > potentials[index]:
                    stack.append((j, chain_length+1))

    print(max_chains)


def get_energy(adjacency_list, potentials, r=1):
    energy = 0.0
    for i, row in enumerate(adjacency_list):
        for index in row:
            energy += (1.0/r)*(potentials[i]-potentials[index])**2

    return energy/2;

## Needs to work for + and x Graph Approximations
'''def save_harmonics(b, l, level, potentials, coordinates, filename):
    with open(filename, 'w') as fout:
        fout.write('Produced by "The Resistance",' + str(datetime.datetime.now()) + '\n')
        fout.write('---------------------------------------------------------\n')
        fout.write('Parameters: b = %d, l = %d, level = %d\n' % (b, l, level))
        fout.write('---------------------------------------------------------\n')
        fout.write('y\tx\tpotential\n')
        for i, coordinate in enumerate(coordinates):
            y, x = coordinate
            fout.write('%d\t%d\t%f\n' % (y, x, potentials[i]))'''
