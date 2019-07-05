import sys

import numpy as np
import matplotlib.pyplot as plt

import shared
import scipy.sparse as sparse
import scipy.sparse.linalg as la
import scipy.interpolate as interpolate

def get_adjacency_list(layout, coordinates):
    adjacency_list = []
    grid_size = np.shape(layout)[0]

    # Determines adjacency list based off neighbors (and # crosswires)
    for coordinate in coordinates:
        row = []
        y, x = coordinate
        if x > 0 and layout[y, x-1] != -1:
            row.append(layout[y, x-1])
        if y > 0 and layout[y-1, x] != -1:
            row.append(layout[y-1, x])
        if x < grid_size-1 and layout[y, x+1] != -1:
            row.append(layout[y, x+1])
        if y < grid_size-1 and layout[y+1, x] != -1:
            row.append(layout[y+1, x])
        adjacency_list.append(row)

    return adjacency_list

def compute_harmonic_function(laplacian, b, crosswires, level):
    ''' Computes the harmonic function for which the left edge of a given
    carpet has potential 0, and the right edge has potential 1'''

    print('Computing Harmonic Function Potentials')

    # Sections of A
    topleft_a = sparse.csr_matrix(laplacian[11:20, 11:20])
    topright_a = sparse.csr_matrix(laplacian[11:20, 31:])
    bottomleft_a = sparse.csr_matrix(laplacian[31:, 11:20])
    bottomright_a = sparse.csr_matrix(laplacian[31:, 31:])

    # Combine Sections with hstack / vstack (CSR cast is due to matrices being turned into COO)
    top_a = sparse.hstack([topleft_a, topright_a])
    bottom_a = sparse.hstack([bottomleft_a, bottomright_a])
    a = sparse.csr_matrix(sparse.vstack([top_a, bottom_a]))
    #print(a.todense())

    # Sections of R
    topleft_r = sparse.csr_matrix(laplacian[11:20, :11])
    topright_r = sparse.csr_matrix(laplacian[11:20, 20:31])
    bottomleft_r = sparse.csr_matrix(laplacian[31:, :11])
    bottomright_r = sparse.csr_matrix(laplacian[31:, 20:31])

    # Combine Sections with hstack / vstack
    top_r = sparse.hstack([topleft_r, topright_r])
    bottom_r = sparse.hstack([bottomleft_r, bottomright_r])
    r = sparse.vstack([top_r, bottom_r])
    #print(r.todense())
    # Set Dirichlet Boundary Conditions (Left / Right Edge)
    dirichlet = np.zeros((22))
    dirichlet[11:] = 1
    b = -r.dot(dirichlet)
    #print(b)

    # Uses a linear algebra solver to compute harmonic function potentials
    potentials = la.spsolve(a, b)

    # Add in boundary conditions for full harmonic function
    potentials = np.insert(potentials, 9, dirichlet[11:])
    potentials = np.insert(potentials, 0, dirichlet[:11])
    print(potentials)
    return potentials


def main():
    # Make printing a bit nicer for visualizing
    np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

    layout = np.array([ [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
                        [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
                        [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
                        [-2, -2, -2, -2, -1, -1, -1, -2, -2, -2, -2, -2],
                        [-2, -2, -2, -2, -1, -1, -1, -2, -2, -2, -2, -2],
                        [-2, -2, -2, -2, -1, -1, -1, -2, -2, -2, -2, -2],
                        [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
                        [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
                        [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2]])

    plt.imshow(layout)
    plt.show()

    coordinates = shared.index_layout(layout)
    print(layout)
    adjacency_list = get_adjacency_list(layout, coordinates)
    print(adjacency_list)
    laplacian = shared.compute_laplacian(adjacency_list)
    potentials = compute_harmonic_function(laplacian, 3, 3, 1)
    #print(potentials)
    harmonic_function = shared.display_harmonic_function(potentials, coordinates, 11, display_type='grid')
    print('resistance', 1/shared.get_energy(adjacency_list, potentials, 1))



if __name__ == '__main__':
    main()
