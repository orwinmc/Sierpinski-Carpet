# General Imports
import sys
import argparse

# Math Imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.sparse.linalg as la

# Shared functions
import shared

def get_grid_size(b, crosswires, level):
    ''' Returns a integer, the number of vertices across for a given Sierpinski
    Carpet'''

    return (crosswires+1) * b**level + 1;

def get_grid_layout(b, l, crosswires, level):
    ''' Constructs a layout for the + graph approximation of Hm,n.  Returns a
    numpy array which contains a -2 at all locations for which there is a vertex
    in the fractal, and -1 in the remaining locations '''

    print('Generating Grid Layout ...')

    if (l!=0 and b%2 != l%2) or b==l:
        print("Invalid Input!")
        return
    else:
        grid_size = (crosswires+1) * b**level + 1
        layout = np.empty((grid_size, grid_size), dtype=object)

        # Removes Corners for the + Symbol
        for y, row in enumerate(layout):
            for x, val in enumerate(row):
                if x%(crosswires+1) == 0 and y%(crosswires+1) == 0:
                    layout[y, x] = -1
                else:
                    layout[y, x] = -2

        # Iterates over each size hole removing it from the layout
        for current_level in range(1, level+1):
            current_grid_size = get_grid_size(b, crosswires, current_level);
            prev_grid_size = get_grid_size(b, crosswires, current_level-1);
            hole_size = l*(prev_grid_size-1)-1;
            for i in range((b-l)//2*(prev_grid_size-1)+1, grid_size, current_grid_size-1):
                for j in range((b-l)//2*(prev_grid_size-1)+1, grid_size, current_grid_size-1):
                    layout[i:i+hole_size, j:j+hole_size] = -1

        return layout

def get_adjacency_list(layout, coordinates, crosswires):
    ''' Constructs the adjacency list for a given SC.  All neighbors are marked
    as being "connected" except those which are on the tips of each + symbol
    (e.g. the top cannot have a horizontal connection)'''

    print('Computing Adjacency List ...')

    adjacency_list = []
    grid_size = np.shape(layout)[0]

    # Determines adjacency list based off neighbors (and # crosswires)
    for coordinate in coordinates:
        row = []
        y, x = coordinate
        if x > 0 and y % (crosswires+1) != 0 and layout[y, x-1] != -1:
            row.append(layout[y, x-1])
        if y > 0 and x % (crosswires+1) != 0 and layout[y-1, x] != -1:
            row.append(layout[y-1, x])
        if x < grid_size-1 and y % (crosswires+1) != 0 and layout[y, x+1] != -1:
            row.append(layout[y, x+1])
        if y < grid_size-1 and x % (crosswires+1) != 0 and layout[y+1, x] != -1:
            row.append(layout[y+1, x])
        adjacency_list.append(row)

    return adjacency_list

def compute_harmonic_function(laplacian, b, crosswires, level):
    ''' Computes the harmonic function for which the left edge of a given
    carpet has potential 0, and the right edge has potential 1'''

    print('Computing Harmonic Function Potentials')

    num_coordinates = np.shape(laplacian)[0]
    num_boundary_points = crosswires*b**level

    # Sections of A
    topleft_a = sparse.csr_matrix(laplacian[num_boundary_points:2*num_boundary_points, num_boundary_points:2*num_boundary_points])
    topright_a = sparse.csr_matrix(laplacian[num_boundary_points:2*num_boundary_points, 3*num_boundary_points:])
    bottomleft_a = sparse.csr_matrix(laplacian[3*num_boundary_points:, num_boundary_points:2*num_boundary_points])
    bottomright_a = sparse.csr_matrix(laplacian[3*num_boundary_points:, 3*num_boundary_points:])

    # Combine Sections with hstack / vstack (CSR cast is due to matrices being turned into COO)
    top_a = sparse.hstack([topleft_a, topright_a])
    bottom_a = sparse.hstack([bottomleft_a, bottomright_a])
    a = sparse.csr_matrix(sparse.vstack([top_a, bottom_a]))

    # Sections of R
    topleft_r = sparse.csr_matrix(laplacian[num_boundary_points:2*num_boundary_points, 0:num_boundary_points])
    topright_r = sparse.csr_matrix(laplacian[num_boundary_points:2*num_boundary_points, 2*num_boundary_points:3*num_boundary_points])
    bottomleft_r = sparse.csr_matrix(laplacian[3*num_boundary_points:, 0:num_boundary_points])
    bottomright_r = sparse.csr_matrix(laplacian[3*num_boundary_points:, 2*num_boundary_points:3*num_boundary_points])

    # Combine Sections with hstack / vstack
    top_r = sparse.hstack([topleft_r, topright_r])
    bottom_r = sparse.hstack([bottomleft_r, bottomright_r])
    r = sparse.vstack([top_r, bottom_r])

    # Set Dirichlet Boundary Conditions (Left / Right Edge)
    dirichlet = np.zeros((2*num_boundary_points))
    dirichlet[num_boundary_points:] = 1
    b = -r.dot(dirichlet)

    # Uses a linear algebra solver to compute harmonic function potentials
    potentials = la.spsolve(a, b)

    # Add in boundary conditions for full harmonic function
    potentials = np.insert(potentials, num_boundary_points, dirichlet[num_boundary_points:])
    potentials = np.insert(potentials, 0, dirichlet[:num_boundary_points])

    return potentials

def main():
    # Make printing a bit nicer for visualizing
    np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

    # Algorithm Parameters (Type -h for usage)
    parser = argparse.ArgumentParser(description='Generates the + Graph Approximations for the Sierpinski Carpet')
    parser.add_argument('-b', default=3, type=int, help='The number of sections to divide the carpet into')
    parser.add_argument('-l', default=1, type=int, help='The number of sections to remove from the carpet center')
    parser.add_argument('-c', '--crosswires', type=int, default=1, help='The number of crosswires')
    parser.add_argument('-a', '--level', type=int, default=3, help='Number of pre-carpet contraction iterations')
    args = parser.parse_args()

    # Begin Computation of Harmonic Function
    print('Generating + Graph Approximation for b=%d, l=%d, crosswires=%d, level=%d ...' % (args.b, args.l, args.crosswires, args.level))
    grid_size = get_grid_size(args.b, args.crosswires, args.level)
    layout = get_grid_layout(args.b, args.l, args.crosswires, args.level)

    # Visualization of Fractal
    shared.display_grid_layout(layout, display_type='matplotlib')

    # Possibly need to clear some memory, insert `del layout` at some point
    coordinates = shared.index_layout(layout)
    adjacency_list = get_adjacency_list(layout, coordinates, args.crosswires)
    laplacian = shared.compute_laplacian(adjacency_list)
    potentials = compute_harmonic_function(laplacian, args.b, args.crosswires, args.level)

    shared.display_harmonic_function(potentials, coordinates, grid_size, display_type='grid')


if __name__ == '__main__':
    main()
