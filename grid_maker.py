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

def get_coordinates(grid_layout):
    '''
        Returns an 2D array for which each row is the (row,col) coordinates
    '''
    #should not work
    num_coordinates = int(np.sum(grid_layout))
    coordinates = np.zeros((num_coordinates, 2), dtype=object)
    counter = 0

    # Left
    for i in range(0, np.shape(grid_layout)[0]):
        if grid_layout[i, 0]:
            coordinates[counter] = (i, 0)
            counter+=1

    # Right
    for i in range(0, np.shape(grid_layout)[0]):
        if grid_layout[i, np.shape(grid_layout)[1]-1]:
            coordinates[counter] = (i, np.shape(grid_layout)[1]-1)
            counter+=1

    #interior
    for y, row in enumerate(grid_layout):
        for x, val in enumerate(row):
            if val and x > 0 and x < np.shape(grid_layout)[1]-1:
                coordinates[counter] = (y, x)
                counter+=1

    print(coordinates)

    return coordinates

def compute_laplacian(indexed_layout, crosswires):
    # Gets coordiantes by checking which are not -1

    num_coordinates = int(np.sum(indexed_layout != -1))

    # Compute Laplacian Matrix
    laplacian = sparse.lil_matrix((num_coordinates, num_coordinates))
    #laplacian = np.zeros((num_coordinates, num_coordinates))
    for y, row in enumerate(indexed_layout):
        for x, val in enumerate(row):
            if val != -1:
                neighbors = 0
                if x > 0 and y % (crosswires+1) != 0 and indexed_layout[y, x-1] != -1:
                    laplacian[indexed_layout[y, x], indexed_layout[y, x-1]] = 1
                    neighbors+=1
                if y > 0 and x % (crosswires+1) != 0 and indexed_layout[y-1, x] != -1:
                    laplacian[indexed_layout[y, x], indexed_layout[y-1, x]] = 1
                    neighbors+=1
                if x < np.shape(indexed_layout)[1]-1 and y % (crosswires+1) != 0 and indexed_layout[y, x+1] != -1:
                    laplacian[indexed_layout[y, x], indexed_layout[y, x+1]] = 1
                    neighbors+=1
                if y < np.shape(indexed_layout)[0]-1 and x % (crosswires+1) != 0 and indexed_layout[y+1, x] != -1:
                    laplacian[indexed_layout[y, x], indexed_layout[y+1, x]] = 1
                    neighbors+=1

                laplacian[indexed_layout[y, x], indexed_layout[y, x]] = -neighbors

    return laplacian

def compute_harmonic_function(laplacian, b, level, c):
    # BAD VARIABLE NAMES, NEEDS TO BE FIXED
    num_boundary_points = c*b**level

    print('laplacian shape', np.shape(laplacian))
    s_n = sparse.csr_matrix(laplacian[num_boundary_points*2: , num_boundary_points*2:])
    print('s_n', np.shape(s_n))
    r_n = sparse.csr_matrix(laplacian[num_boundary_points*2: , :num_boundary_points*2])
    print('r_n', np.shape(r_n))
    u_n = np.zeros((2*num_boundary_points))
    u_n[num_boundary_points:] = 1
    print(u_n)
    print('u_n', np.shape(u_n))
    val = -r_n.dot(u_n)
    print(np.shape(val))

    w_n = la.spsolve(s_n, -r_n.dot(u_n))

    w_n = np.concatenate((u_n, w_n), axis=0)
    print(w_n)
    #print(w_n*59364)

    return w_n

def compute_energy(laplacian, u_n):
    energy = 0

    for y in range(np.shape(laplacian)[0]):
        for x in range(np.shape(laplacian)[1]):
            if laplacian[y, x] == 1:
                energy += (u_n[y]-u_n[x])**2

    return energy / 2

def laid_out_harmonic(u_n, indexed_layout):
    image = np.full(np.shape(indexed_layout), 0, dtype=float)
    #print(image)
    for y in range(np.shape(indexed_layout)[0]):
        for x in range(np.shape(indexed_layout)[1]):
            if indexed_layout[y, x] != -1:
                image[y, x] = u_n[indexed_layout[y, x]]
            else:
                image[y, x] = None

    X = range(np.shape(indexed_layout)[0])
    Y = range(np.shape(indexed_layout)[0])
    #H = image[Y, X]
    print(X)
    #print(H)
    #print(image)
    #plt.imshow(image)
    plt.contour(X, Y, image, levels=np.linspace(0,1,num=100))
    plt.show()

def main():
    # Tries to make printing better (fails)
    np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

    # Input Values
    b = 3
    l = 1
    level = 3
    crosswires = 7

    grid_layout = get_grid_layout(b, l, level, crosswires)
    plt.imshow(grid_layout)
    plt.show()

    print('Computing \'indexed_layout\' ...')
    indexed_layout = get_indexed_layout(grid_layout)
    #print(indexed_layout)

    print('Computing \'coordinates\' ...')
    coordinates = get_coordinates(grid_layout)
    print(coordinates)

    print('Number of Vertices: %d' % (np.shape(coordinates)[0]))

    print('Computing \'laplacian\' ...')
    laplacian = compute_laplacian(indexed_layout, crosswires)
    #pic_laplacian = laplacian.todense()
    #pic_laplacian[pic_laplacian == 0] = np.nan
    #print(laplacian)
    #plt.imshow(laplacian.todense())
    #data_masked = np.ma.masked_where(data == -1, pic_laplacian)
    plt.imshow(laplacian.todense())
    plt.show()

    u_n = compute_harmonic_function(laplacian, b, level, crosswires)
    #print('Computing \'energy\' ...')
    #energy = compute_energy(laplacian, u_n)
    #print(1/energy)

    with open('123.tsv', 'w') as fin:
        fin.write('y\tx\tpotential\n')
        for i, potential in enumerate(u_n):
            fin.write('%f\t%f\t%f\n' % (coordinates[i][0], coordinates[i][1], potential))

    laid_out_harmonic(u_n, indexed_layout)


if __name__ == '__main__':
    main()
