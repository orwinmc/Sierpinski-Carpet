import sys
import argparse
from tqdm import tqdm

# Math Imports
import numpy as np
import matplotlib.pyplot as plt

# Shared functions
import shared

def get_coordinates(b, l, level, resolution):
    # The number of vertices in the graphical approximation
    num_vertices = (b**2-l**2)**level * resolution**2

    # Holds all Coordinates
    coordinates = np.zeros(num_vertices, dtype=[('y', 'i8'), ('x', 'i8')])

    # "Base Case"
    for i in range(resolution):
        for j in range(resolution):
            coordinates[i*resolution+j] = (i,j)

    # Next index which needs to be filled in the coordinate list
    current = resolution**2
    # Horizontal width of each square
    shift = resolution
    # The number of vertices that are being copied from previous version
    window = resolution**2

    for a in range(level):
        for i in range(b):
            for j in range(b):
                if i < (b-l)/2 or i >= (b+l)/2 or j < (b-l)/2 or j >= (b+l)/2:
                    if i>0 or j>0:
                        for k in range(window):
                            coordinates[current+k] = (coordinates[k][0]+i*shift, coordinates[k][1]+j*shift)
                        current+=window

        window*=b**2-l**2
        shift*=b

    coordinates.sort()
    return coordinates

def get_adjacency_list(coordinates):
    ''' Constructs the adjacency list for a given SC'''

    print('Computing Adjacency List ...')

    adjacency_list = []

    for i, coord in tqdm(enumerate(coordinates), total=len(coordinates)):
        row = []

        adjacent_vertices = [np.array((coord[0]-1, coord[1]), dtype=coord.dtype),
                            np.array((coord[0]+1, coord[1]), dtype=coord.dtype),
                            np.array((coord[0], coord[1]-1), dtype=coord.dtype),
                            np.array((coord[0], coord[1]+1), dtype=coord.dtype)]

        for j,p in enumerate(adjacent_vertices):
            closest_index = np.searchsorted(coordinates, p)
            #print(closest_index)
            if closest_index < len(coordinates) and coordinates[closest_index] == p:
                row.append(closest_index)

        adjacency_list.append(row)

    return adjacency_list

def get_edge_length(b, level, resolution):
    return b**level*resolution

def top_to_bottom_potentials(b, level, resolution, laplacian):
    edge_length = get_edge_length(b, level, resolution)

    # Calculate Boundary Indices
    boundary_indices = []
    boundary_indices.extend(range(edge_length))
    boundary_indices.extend(range(laplacian.shape[0]-edge_length,laplacian.shape[0]))

    # Set Dirichlet Boundary
    boundary = np.zeros((2*edge_length))
    boundary[edge_length:] = 1

    # Compute Harmonic Function
    potentials = shared.compute_harmonic_function(laplacian, boundary_indices, boundary)
    harmonic_function = shared.display_harmonic_function(potentials, coordinates, edge_length, display_type='grid')

def main():
    # Make printing a bit nicer for visualizing
    np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

    # Algorithm Parameters (Type -h for usage)
    parser = argparse.ArgumentParser(description='Generates the + Graph Approximations for the Sierpinski Carpet')
    parser.add_argument('-b', type=int, default=3, help='The number of sections to divide the carpet into')
    parser.add_argument('-l', type=int, default=1, help='The number of sections to remove from the carpet center')
    parser.add_argument('-r', '--resolution', type=int, default=1, help='The number of vertices per square')
    parser.add_argument('-a', '--level', type=int, default=2, help='Number of pre-carpet contraction iterations')
    args = parser.parse_args()

    # Begin Computation of Harmonic Function
    print('Generating Basic Graph Approximation for b=%d, l=%d, resolution=%d, level=%d ...' % (args.b, args.l, args.resolution, args.level))

    coordinates = get_coordinates(args.b, args.l, args.level, args.resolution)
    adjacency_list = get_adjacency_list(coordinates)
    laplacian = shared.compute_laplacian(adjacency_list)
    
    top_to_bottom_potentials(args.b, args.level, args.resolution, laplacian)

if __name__ == '__main__':
    main()
