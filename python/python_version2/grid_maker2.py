import numpy as np
import matplotlib.pyplot as plt
import sys # used to eliminate printing truncation
from scipy import sparse
import scipy.sparse.linalg as la

# Constructs a 2D array of booleans, true = a given vertex is inside the SC
def get_grid_layout(b, l, level, c):
    '''
        Generates a 2D array that looks something like this

        0,1,1,1,0
        1,1,1,1,1
        1,1,1,1,1
        1,1,1,1,1
        0,1,1,1,0

        Works on multiple levels!
    '''
    if (l!=0 and b%2 != l%2) or b==l:
        print("Invalid Input!")
        return

    num_vertices = (c+1) * b**level + 1
    sc = np.ones((num_vertices, num_vertices))

    for y, row in enumerate(sc):
        for x, val in enumerate(sc):
            if x%(c+1) == 0 and y%(c+1) == 0:
                sc[y, x] = 0

    for current_level in range(1, level+1):
        vertices_current = (c+1)*b**current_level + 1
        prev_vertices_current = (c+1)*b**(current_level-1) + 1
        hole_size = l*(c+1)*b**(current_level-1)-1
        for i in range((b-l)//2*(prev_vertices_current-1)+1, num_vertices, vertices_current-1):
            for j in range((b-l)//2*(prev_vertices_current-1)+1, num_vertices, vertices_current-1):
                sc[i:i+hole_size, j:j+hole_size] = 0

    return sc

def get_indexed_layout(grid_layout):
    '''
        Generates a 2D array that looks something like this

        -1,0,1,2,-1
        3,4,5,6,7
        8,9,10,11,12
        13,14,15,16,17
        -1,18,19,20,-1

        Each number represents the row of vector which will hold the laplacian
    '''
    indexed_layout = np.full(np.shape(grid_layout), -1, dtype=object)
    counter = 0

    # Left
    for i in range(0, np.shape(grid_layout)[0]):
        if grid_layout[i, 0]:
            indexed_layout[i, 0] = counter
            counter+=1

    # Right
    for i in range(0, np.shape(grid_layout)[0]):
        if grid_layout[i, np.shape(grid_layout)[1]-1]:
            indexed_layout[i, np.shape(grid_layout)[1]-1] = counter
            counter+=1

    # Interior
    for y, row in enumerate(grid_layout):
        for x, val in enumerate(row):
            if val and indexed_layout[y, x] == -1:
                indexed_layout[y,x] = counter
                counter+=1

    #print(indexed_layout)

    return indexed_layout

def compute_laplacian(grid_layout, indexed_layout, crosswires):
    num_coordinates = int(np.sum(indexed_layout != -1))

    # Compute Laplacian Matrix
    laplacian = sparse.lil_matrix((num_coordinates, num_coordinates))

    index = 0
    stack = [(0,1)]
    test = 0
    while len(stack) > 0:
        print(test)
        test+=1
        vertex = stack.pop()
        neighbors = 0
        if grid_layout[vertex[0],vertex[1]] == 2:
            continue
        else:
            grid_layout[vertex[0], vertex[1]] = 2
        if vertex[1] > 0 and vertex[0] % (crosswires+1) != 0 and indexed_layout[vertex[0], vertex[1]-1] != -1:
            laplacian[indexed_layout[vertex[0], vertex[1]], indexed_layout[vertex[0], vertex[1]-1]] = 1
            neighbors+=1
            stack.append((vertex[0],vertex[1]-1))
        if vertex[0] > 0 and vertex[1] % (crosswires+1) != 0 and indexed_layout[vertex[0]-1, vertex[1]] != -1:
            laplacian[indexed_layout[vertex[0], vertex[1]], indexed_layout[vertex[0]-1, vertex[1]]] = 1
            neighbors+=1
            stack.append((vertex[0]-1,vertex[1]))
        if vertex[1] < np.shape(indexed_layout)[1]-1 and vertex[0] % (crosswires+1) != 0 and indexed_layout[vertex[0], vertex[1]+1] != -1:
            laplacian[indexed_layout[vertex[0], vertex[1]], indexed_layout[vertex[0], vertex[1]+1]] = 1
            neighbors+=1
            stack.append((vertex[0],vertex[1]+1))
        if vertex[0] < np.shape(indexed_layout)[0]-1 and vertex[1] % (crosswires+1) != 0 and indexed_layout[vertex[0]+1, vertex[1]] != -1:
            laplacian[indexed_layout[vertex[0], vertex[1]], indexed_layout[vertex[0]+1, vertex[1]]] = 1
            neighbors+=1
            stack.append((vertex[0]+1,vertex[1]))

        #indexed_layout[(vertex[0],vertex[1])] =

        laplacian[indexed_layout[vertex[0], vertex[1]], indexed_layout[vertex[0], vertex[1]]] = -neighbors

    return laplacian


def main():
    # Input Values
    b = 3
    l = 1
    level = 6
    crosswires = 1

    print('Computing \'grid_layout\' ...')
    grid_layout = get_grid_layout(b, l, level, crosswires)
    #plt.imshow(grid_layout)
    #plt.show()

    print('Computing \'indexed_layout\' ...')
    indexed_layout = get_indexed_layout(grid_layout)
    print(indexed_layout)

    print('Computing \'laplacian\' ...')
    laplacian = compute_laplacian(grid_layout, indexed_layout, crosswires)
    print(laplacian)
    #plt.imshow(laplacian.todense())
    #plt.show()


if __name__ == '__main__':
    main()
