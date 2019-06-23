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

def display_harmonic_function(potentials, coordinates, grid_size, display_type='grid', num_contours=200):
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

        plt.contour(ys, xs, harmonic_function, levels=np.linspace(0, 1, num=num_contours))
        plt.colorbar()
        plt.show()

    return harmonic_function

def index_layout(layout):
    ''' Enumerates the layout of the given fractal.  At each location a -2 is
    replaced by the index of the coordinate. Boundaries are indexed first in
    counterclockwise order (but smallest to biggest) starting with the
    left edge. '''

    print('Indexing the given layout ...')

    coordinates = []
    grid_size = np.shape(layout)[0]
    current_index = 0

    # Enumerates Left Edge
    for y in range(grid_size):
        if layout[y, 0] == -2:
            layout[y, 0] = current_index
            coordinates.append((y, 0))
            current_index += 1

    # Enumerates Bottom Edge
    for x in range(grid_size):
        if layout[grid_size-1, x] == -2:
            layout[grid_size-1, x] = current_index
            coordinates.append((grid_size-1, x))
            current_index += 1

    # Enumerates Right Edge
    for y in range(grid_size):
        if layout[y, grid_size-1] == -2:
            layout[y, grid_size-1] = current_index
            coordinates.append((y, grid_size-1))
            current_index += 1

    # Enumerates Bottom Edge
    for x in range(grid_size):
        if layout[0, x] == -2:
            layout[0, x] = current_index
            coordinates.append((0, x))
            current_index += 1

    # Enumerates remaining vertices
    for y,row in tqdm(enumerate(layout), total=np.shape(layout)[0]):
        for x,val in enumerate(row):
            if val == -2:
                layout[y, x] = current_index
                coordinates.append((y, x))
                current_index += 1

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

    return sparse.csr_matrix(laplacian)

def compute_harmonic_function(laplacian, boundary_indices, boundary):
    print('Computing Harmonic Function Potentials ...')
    num_boundary_points = len(boundary_indices)
    num_computed_points = laplacian.get_shape()[0]-num_boundary_points
    total_points = num_boundary_points + num_computed_points

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

    perm = sparse.csr_matrix(perm)

    reordered_laplacian = perm.dot(laplacian.dot(perm))

    a = reordered_laplacian[num_boundary_points:, num_boundary_points:]
    r = reordered_laplacian[num_boundary_points:, :num_boundary_points]
    b = -r.dot(boundary)

    # Uses a linear algebra solver to compute harmonic function potentials
    potentials = la.spsolve(a, b)

    return potentials


def max_edges(adjacency_list, potentials, coordinates, grid_size):
    ''' Finds the maximum edges in a given fractal (based on the associated
    adjacency matrix).  Returns the coordinates of these edges in a list
    (Should probably just return the indices based on coordinates****)'''

    max_diff = 0
    max_edges = []

    for i, row in enumerate(adjacency_list):
        for index in row:
            if i < index:
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


def get_energy(adjacency_list, potentials, r=1):
    energy = 0.0
    for i, row in enumerate(adjacency_list):
        for index in row:
            energy += (1.0/r)*(potentials[i]-potentials[index])**2

    return energy/2;


def save_harmonics(b, l, level, potentials, coordinates, filename):
    with open(filename, 'w') as fout:
        fout.write('Produced by "The Resistance",' + str(datetime.datetime.now()) + '\n')
        fout.write('---------------------------------------------------------\n')
        fout.write('Parameters: b = %d, l = %d, level = %d\n' % (b, l, level))
        fout.write('---------------------------------------------------------\n')
        fout.write('y\tx\tpotential\n')
        for i, coordinate in enumerate(coordinates):
            y, x = coordinate
            fout.write('%d\t%d\t%f\n' % (y, x, potentials[i]))
