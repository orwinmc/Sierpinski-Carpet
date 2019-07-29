import sys

import numpy as np
import matplotlib.pyplot as plt

import shared
import scipy.sparse as sparse
import scipy.sparse.linalg as la
import scipy.interpolate as interpolate
from tqdm import tqdm

def get_adjacency_list(layout, coordinates, crosswires):
    ''' Constructs the adjacency list for a given SC.  All neighbors are marked
    as being "connected" except those which are on the tips of each + symbol
    (e.g. the top cannot have a horizontal connection)'''

    print('Computing Adjacency List ...')

    adjacency_list = []
    grid_size = np.shape(layout)[0]

    # Determines adjacency list based off neighbors (and # crosswires)
    for coordinate in tqdm(coordinates, total=len(coordinates)):
        row = []
        y, x = coordinate
        if x > 0 and y % (crosswires+2) != 0 and y % (crosswires+2) != crosswires+1 and layout[y, x-1] != -1:
            row.append(layout[y, x-1])
        if y > 0 and x % (crosswires+2) != 0 and x % (crosswires+2) != crosswires+1 and layout[y-1, x] != -1:
            row.append(layout[y-1, x])
        if x < grid_size-1 and y % (crosswires+2) != 0 and y % (crosswires+2) != crosswires+1 and layout[y, x+1] != -1:
            row.append(layout[y, x+1])
        if y < grid_size-1 and x % (crosswires+2) != 0 and x % (crosswires+2) != crosswires+1 and layout[y+1, x] != -1:
            row.append(layout[y+1, x])
        adjacency_list.append(row)

    return adjacency_list

def compute_harmonic_function(laplacian, b, crosswires, level):
    ''' Computes the harmonic function for which the left edge of a given
    carpet has potential 0, and the right edge has potential 1'''

    print('Computing Harmonic Function Potentials')

    # Sections of A
    topleft_a = sparse.csr_matrix(laplacian[9:18, 9:18])
    topright_a = sparse.csr_matrix(laplacian[9:18, 27:])
    bottomleft_a = sparse.csr_matrix(laplacian[27:, 9:18])
    bottomright_a = sparse.csr_matrix(laplacian[27:, 27:])

    # Combine Sections with hstack / vstack (CSR cast is due to matrices being turned into COO)
    top_a = sparse.hstack([topleft_a, topright_a])
    bottom_a = sparse.hstack([bottomleft_a, bottomright_a])
    a = sparse.csr_matrix(sparse.vstack([top_a, bottom_a]))
    #plt.imshow(a.todense())
    #plt.show()

    # Sections of R
    topleft_r = sparse.csr_matrix(laplacian[9:18, :9])
    topright_r = sparse.csr_matrix(laplacian[9:18, 18:27])
    bottomleft_r = sparse.csr_matrix(laplacian[27:, :9])
    bottomright_r = sparse.csr_matrix(laplacian[27:, 18:27])

    # Combine Sections with hstack / vstack
    top_r = sparse.hstack([topleft_r, topright_r])
    bottom_r = sparse.hstack([bottomleft_r, bottomright_r])
    r = sparse.vstack([top_r, bottom_r])

    #plt.imshow(r.todense())
    #plt.show()

    # Set Dirichlet Boundary Conditions (Left / Right Edge)
    dirichlet = np.zeros((18))
    dirichlet[9:] = 1
    b = -r.dot(dirichlet)

    # Uses a linear algebra solver to compute harmonic function potentials
    potentials = la.spsolve(a, b)

    # Add in boundary conditions for full harmonic function
    potentials = np.insert(potentials, 9, dirichlet[9:])
    potentials = np.insert(potentials, 0, dirichlet[:9])
    print(potentials)
    return potentials



def main():
    print('sdf')
    # Make printing a bit nicer for visualizing
    np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)
    #print('asdfsadf')

    base_layout = np.array([[-1, -2, -2, -2, -1],
                        [-2, -2, -2, -2, -2],
                        [-2, -2, -2, -2, -2],
                        [-2, -2, -2, -2, -2],
                        [-1, -2, -2, -2, -1]])

    blank_layout = np.full((5,5), -1)

    x = np.hstack((base_layout, base_layout, base_layout))
    y = np.hstack((base_layout, blank_layout, base_layout))
    z = np.hstack((base_layout, base_layout, base_layout))

    layout = np.vstack((x, y, z))
    print(layout)
    plt.imshow(layout)
    plt.show()

    coordinates = shared.index_layout(layout)
    print(layout)


    adjacency_list = get_adjacency_list(layout, coordinates, 3)
    print(adjacency_list)
    laplacian = shared.compute_laplacian(adjacency_list)
    #plt.imshow(laplacian.todense())
    #print(laplacian.todense())
    #plt.show()
    potentials = compute_harmonic_function(laplacian, 3, 3, 1)
    #print(potentials)
    harmonic_function = shared.display_harmonic_function(potentials, coordinates, 15, display_type='grid')
    print('resistance', 1/shared.get_energy(adjacency_list, potentials, 1))

if __name__ == '__main__':
    main()
