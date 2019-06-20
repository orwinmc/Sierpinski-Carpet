import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.interpolate as sci_terp

def read_harmonic_function(filename):
    offset = 4
    coordinates = []
    potentials = []

    with open(filename, 'r') as fin:
        for line_num, line in enumerate(fin):
            if line_num > offset:
                y, x, potential = tuple(map(float, line.split()))
                coordinates.append((int(y), int(x)))
                potentials.append(potential)

    return np.array(coordinates), np.array(potentials)

def get_grid_size(b, level, c):
    return (c+1) * b**level + 1;

def get_indexed_layout(coordinates, potentials, grid_size):
    indexed_layout = np.full((grid_size, grid_size), -1)

    for i, coordinate in enumerate(coordinates):
        indexed_layout[coordinate[0], coordinate[1]] = i

    #print(indexed_layout)

    return indexed_layout

def get_harmonic_layout(coordinates, potentials, grid_size):
    harmonic_layout = np.full((grid_size, grid_size), -1, dtype=float)

    for i, coordinate in enumerate(coordinates):
        harmonic_layout[coordinate[0], coordinate[1]] = potentials[i]

    #print(harmonic_layout)
    plt.imshow(harmonic_layout)
    plt.show()

    return harmonic_layout

def get_adjacency_list(indexed_layout, coordinates, crosswires):
    adjacency_list = []
    for i, coordinate in enumerate(coordinates):
        row = []
        y, x = coordinate
        if x > 0 and y % (crosswires+1) != 0 and indexed_layout[y, x-1] != -1:
            row.append(indexed_layout[y, x-1])
        if y > 0 and x % (crosswires+1) != 0 and indexed_layout[y-1, x] != -1:
            row.append(indexed_layout[y-1, x])
        if x < np.shape(indexed_layout)[1]-1 and y % (crosswires+1) != 0 and indexed_layout[y, x+1] != -1:
            row.append(indexed_layout[y, x+1])
        if y < np.shape(indexed_layout)[0]-1 and x % (crosswires+1) != 0 and indexed_layout[y+1, x] != -1:
            row.append(indexed_layout[y+1, x])
        adjacency_list.append(row)

    return adjacency_list

def find_max(adjacency_list, potentials, coordinates, grid_size):
    max = 0
    left = (-1, -1)
    right = (-1, -1)
    print(grid_size)
    for i, row in enumerate(adjacency_list):
        if coordinates[i][0] <= (grid_size-1)/2 and coordinates[i][1] <= (grid_size-1)/2:
            for index in row:
                if abs(potentials[index]-potentials[i])-max > 0.0000001:
                    max = abs(potentials[index]-potentials[i])
                    if potentials[index] < potentials[i]:
                        left = coordinates[index]
                        right = coordinates[i]
                    else:
                        left = coordinates[i]
                        right = coordinates[index]

    return max, left, right

def get_max_cell(harmonic_layout, left, b, level, crosswires):
    cell_size = get_grid_size(b, 1, crosswires)
    top_left_corner = ((cell_size-1)*(left[0]//(cell_size-1)), (cell_size-1)*(left[1]//(cell_size-1)))
    print(top_left_corner)

    cell = harmonic_layout[top_left_corner[0]:top_left_corner[0]+cell_size, top_left_corner[1]:top_left_corner[1]+cell_size]

    plt.imshow(cell, vmin=-1, vmax=1)
    plt.show()

    return cell

def generate_interpolation(cell, b, c):
    points = []
    potentials = []

    cell_size = np.shape(cell)[0]

    # Top Left Corner
    cell[0, 0] = cell[1, 0] + (cell[0, 1]-cell[1, 1])

    # Top Right Corner
    cell[0, cell_size-1] = cell[1, cell_size-1] + (cell[0, cell_size-2]-cell[1, cell_size-2])

    # Bottom Left Corner
    cell[cell_size-1, 0] = cell[cell_size-2, 0] + (cell[cell_size-1, 1]-cell[cell_size-2, 1])

    # Bottom Right Corner
    cell[cell_size-1, cell_size-1] = cell[cell_size-2, cell_size-1] + (cell[cell_size-1, cell_size-2]-cell[cell_size-2, cell_size-2])

    for y, row in enumerate(cell):
        for x, potential in enumerate(row):
            if potential != -1:
                points.append((y,x))
                potentials.append(potential)

    print(points)
    print(potentials)
    f = sci_terp.LinearNDInterpolator(np.array(points), np.array(potentials), fill_value=-1)

    table = np.full((109, 109), -1, dtype=float)
    for i, y in enumerate(np.linspace(0, cell_size-1, num=109)):
        for j, x in enumerate(np.linspace(0, cell_size-1, num=109)):
            table[i, j] = f(y, x)

    plt.imshow(table, vmin=-1, vmax=1)
    plt.show()


def main():
    np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)
    # Input Values
    b = 3
    l = 1
    crosswires = 1
    level = 3
    grid_size = get_grid_size(b, level, crosswires)

    coordinates, potentials = read_harmonic_function('../../data/plus/3_1_1/plusdata_3_1_1_level3.dat')
    indexed_layout = get_indexed_layout(coordinates, potentials, grid_size)
    adjacency_list = get_adjacency_list(indexed_layout, coordinates, crosswires)
    max, left, right = find_max(adjacency_list, potentials, coordinates, grid_size)
    print(max, left, right)
    harmonic_layout = get_harmonic_layout(coordinates, potentials, grid_size)

    cell = get_max_cell(harmonic_layout, left, b, level, crosswires)

    generate_interpolation(cell, b, crosswires)


if __name__ == '__main__':
    main()
