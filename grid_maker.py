import numpy as np
import sys # used to eliminate printing truncation

# Constructs a 2D array of booleans, true = a given vertex is inside the SC

def get_grid_layout(b, l, level, num_crosswires):
    '''
        Generates a 2D array that looks something like this

        0,1,1,1,0
        1,1,1,1,1
        1,1,1,1,1
        1,1,1,1,1
        0,1,1,1,0

        Works on multiple levels!
    '''
    if b%2 != l%2 or b==l:
        print("Invalid Input!")
        return

    # Build Basic Building Block for SC
    base = np.ones((num_crosswires+2, num_crosswires+2), dtype=bool)
    base[0,0] = 0
    base[0,num_crosswires+1] = 0
    base[num_crosswires+1,0] = 0
    base[num_crosswires+1,num_crosswires+1] = 0

    # Iterate through each level blowing the grid up
    for i in range(level):
        filled_center = base
        empty_center = base

        # Extend Rows of Building Block
        for j in range(1, b):
            filled_center = np.concatenate((filled_center, base), axis=1)
            if j < (b-l)/2 or j >= (b-l)/2+l:
                empty_center = np.concatenate((empty_center, base), axis=1)
            else:
                empty_center = np.concatenate((empty_center, np.zeros(np.shape(base), dtype=bool)), axis=1)

        new_base = filled_center

        # Stack Rows to Achieve new
        for j in range(1, b):
            if j < (b-l)/2 or j >= (b-l)/2+l:
                new_base = np.concatenate((new_base, filled_center), axis=0)
            else:
                new_base = np.concatenate((new_base, empty_center), axis=0)

        base = new_base

    return base

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
    for y, row in enumerate(grid_layout):
        for x, val in enumerate(row):
            if val:
                indexed_layout[y,x] = counter
                counter+=1

    return indexed_layout

def get_coordinates(grid_layout):
    '''
        Returns an 2D array for which each row is the (row,col) coordinates
    '''
    num_coordinates = int(np.sum(grid_layout))
    coordinates = np.zeros((num_coordinates, 2), dtype=object)
    counter = 0
    for y, row in enumerate(grid_layout):
        for x, val in enumerate(row):
            if val:
                coordinates[counter] = (y, x)
                counter+=1

    return coordinates

def compute_laplacian(indexed_layout, crosswires):
    # Gets coordiantes by checking which are not -1
    num_coordinates = int(np.sum(indexed_layout != -1))

    # Compute Laplacian Matrix
    laplacian = np.zeros((num_coordinates, num_coordinates), dtype=object)
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

def main():
    # Tries to make printing better (fails)
    np.set_printoptions(threshold=sys.maxsize)

    # Input Values
    b = 3
    l = 1
    level = 1
    crosswires = 2

    grid_layout = get_grid_layout(b, l, level, crosswires)

    print('Computing \'indexed_layout\' ...')
    indexed_layout = get_indexed_layout(grid_layout)
    print(indexed_layout)

    print('Computing \'coordinates\' ...')
    coordinates = get_coordinates(grid_layout)
    print(coordinates)


    print('Number of Vertices: %d' % (np.shape(coordinates)[0]))

    print('Computing \'laplacian\' ...')
    laplacian = compute_laplacian(indexed_layout, crosswires)
    print(laplacian[21])


if __name__ == '__main__':
    main()
