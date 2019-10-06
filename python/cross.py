# General Imports
from __future__ import print_function
import sys
import argparse
from tqdm import tqdm

# Math Imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.sparse.linalg as la

# Shared functions
import shared

def get_grid_size(b, level):
    ''' Return the number of rows in the 2D which holds the Graphical
    Approximation (One can also think of this as the highest y value of the
    coordinates list + 1). '''

    return 2 * b**level + 1

def get_edge_length(b, level):
    ''' Return the number of vertices which are on the border of the Graphical
    Approximation. '''

    grid_size = get_grid_size(b, level)
    return grid_size//2+1

def get_grid_layout(b, l, level):
    ''' Returns a 2D Numpy array of dtype=object.  Every location either has the
    value -1 or -2.  If a array location has -2 that means a vertex is present in
    the graphical approximation.  Alternatively -1 represents a hole in the
    approximation.'''

    print('Generating Grid Layout ...')

    if (l!=0 and b%2 != l%2) or b==l:
        print("Invalid Input!")
        sys.exit(1)
    else:
        grid_size = get_grid_size(b, level)
        layout = np.empty((grid_size, grid_size), dtype=object)

        # Removes non-diagonal holes formed by each x
        for y, row in enumerate(layout):
            for x, val in enumerate(row):
                if x%2 != y%2:
                    layout[y, x] = -1
                else:
                    layout[y, x] = -2

        # Iterates over each size hole removing it from the layout
        for current_level in range(1, level+1):
            current_grid_size = get_grid_size(b, current_level)
            prev_grid_size = get_grid_size(b, current_level-1)
            hole_size = l*(prev_grid_size-1)-1
            for i in range((b-l)//2*(prev_grid_size-1)+1, grid_size, current_grid_size-1):
                for j in range((b-l)//2*(prev_grid_size-1)+1, grid_size, current_grid_size-1):
                    layout[i:i+hole_size, j:j+hole_size] = -1

    return layout

def get_adjacency_list(layout, coordinates):
    ''' Returns a generic python list which acts as an adjaceny list for the
    coordinates provided.  For the x Graphical Approximation, vertices are
    connected if they are diagonal from each other. '''

    print('Computing Adjacency List ...')

    adjacency_list = []
    grid_size = np.shape(layout)[0]

    # Determines adjacency list based off neighbors (diagonals)
    for coordinate in tqdm(coordinates, total=len(coordinates)):
        row = []
        y, x = coordinate
        if x > 0 and y > 0 and layout[y-1, x-1] != -1:
            row.append(layout[y-1, x-1])
        if x < grid_size-1 and y > 0 and layout[y-1, x+1] != -1:
            row.append(layout[y-1, x+1])
        if x > 0 and y < grid_size-1 and layout[y+1, x-1] != -1:
            row.append(layout[y+1, x-1])
        if x < grid_size-1 and y < grid_size-1 and layout[y+1, x+1] != -1:
            row.append(layout[y+1, x+1])
        adjacency_list.append(row)

    return adjacency_list

def left_to_right_potentials(b, level, coordinates, laplacian):
    ''' Generates the harmonic function which is entirely 1 on the left side,
    and 0 on the right side.  In order to do this, the associated indices of the
    boundary are computed.  The potentials computed are returned'''

    grid_size = get_grid_size(b, level)
    edge_length = get_edge_length(b, level)

    # Get Boundary Indices
    boundary_indices = []
    boundary_indices.extend(range(edge_length))
    boundary_indices.extend(range(2*edge_length-2, 3*edge_length-2))

    # Dirichlet Boundary Conditions
    boundary = np.zeros((2*edge_length))
    boundary[:edge_length] = 1

    potentials = shared.compute_harmonic_function(laplacian, boundary_indices, boundary)
    harmonic_function = shared.display_harmonic_function(potentials, coordinates, grid_size, display_type='grid')

    return potentials

def main():
    ''' Executed with `python cross.py`.  This takes parameters from the user
    and generates the associated harmonic function potentials. '''

    # Make printing a bit nicer for visualizing
    np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

    # Algorithm Parameters (Type -h for usage)
    parser = argparse.ArgumentParser(description='Generates the x Graph Approximations for the Sierpinski Carpet')
    parser.add_argument('-b', default=3, type=int, help='The number of sections to divide the carpet into')
    parser.add_argument('-l', default=1, type=int, help='The number of sections to remove from the carpet center')
    parser.add_argument('-a', '--level', type=int, default=3, help='Number of pre-carpet contraction iterations')
    args = parser.parse_args()

    # Begin Computation of Harmonic Function
    print('Generating x Graph Approximation for b=%d, l=%d, level=%d ...' % (args.b, args.l, args.level))
    grid_size = get_grid_size(args.b, args.level)
    layout = get_grid_layout(args.b, args.l, args.level)

    # Visualization of Fractal
    shared.display_grid_layout(layout, display_type='matplotlib')

    # Possibly need to clear some memory, insert `del layout` at some point
    coordinates = shared.index_layout(layout)
    adjacency_list = get_adjacency_list(layout, coordinates)
    laplacian = shared.compute_laplacian(adjacency_list)

    potentials = left_to_right_potentials(args.b, args.level, coordinates, laplacian)

if __name__ == '__main__':
    main()
