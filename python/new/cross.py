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
    return 2 * b**level + 1

def get_grid_layout(b, l, level):
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

def main():
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

    edge_length = grid_size//2+1
    boundary_indices = []
    boundary_indices.extend(range(edge_length))
    boundary_indices.extend(range(2*edge_length-2, 3*edge_length-2))
    #print(boundary_indices)

    boundary = np.zeros((2*edge_length))
    boundary[edge_length:] = 1
    #print(boundary)

    potentials = shared.compute_harmonic_function(laplacian, boundary_indices, boundary)
    harmonic_function = shared.display_harmonic_function(potentials, coordinates, grid_size, display_type='grid')

    print('resistance', 1/shared.get_energy(adjacency_list, potentials))

    #potentials = compute_harmonic_function(laplacian, args.b, args.level)

    #shared.display_harmonic_function(potentials, coordinates, grid_size, display_type='grid')
    max_edges = shared.max_edges(adjacency_list, potentials, coordinates, grid_size)
    print(coordinates[max_edges[0,0]], coordinates[max_edges[0,1]])
    print(abs(potentials[max_edges[0,0]]-potentials[max_edges[0,1]]))

    '''filename = '../../data/cross/'+str(args.b)+'_'+str(args.l)+'/crossdata_'+str(args.b)+'_'+str(args.l)+'_level'+str(args.level)+'.dat'
    shared.save_harmonics(args.b, args.l, args.level, potentials, coordinates, filename)'''

if __name__ == '__main__':
    main()
