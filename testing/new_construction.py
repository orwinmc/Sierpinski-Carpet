import numpy as np
import matplotlib.pyplot as plt

def get_grid_size(b, level, crosswires):
    return (crosswires+1) * (b**level) + 1

def get_coordinates(b, l, level, crosswires):
    # Construct Base Object
    coordinates = []
    for y in range(crosswires+2):
        for x in range(crosswires+2):
                coordinates.append((y,x))

    # Eliminate Corners
    coordinates.remove((0, 0))
    coordinates.remove((crosswires+1, 0))
    coordinates.remove((0, crosswires+1))
    coordinates.remove((crosswires+1, crosswires+1))

    # Merge Carpets
    for i in range(level):
        print(i)
        new_coordinates = set()

        # Duplicate Base
        for j in range(b):
            for k in range(b):
                print(j, k)
                if j==0 and k==0:
                    continue;
                elif j < (b-l)/2 or j >= (b+l)/2 or k < (b-l)/2 or k >= (b+l)/2:
                    #print(j, k)
                    for coordinate in coordinates:

                        new_coordinates.append((coordinate[0]+j*(get_grid_size(b, i, crosswires)-1) , coordinate[1]+k*(get_grid_size(b, i, crosswires)-1)))

        coordinates = coordinates.union(new_coordinates)

    return coordinates


def main():
    # Input Values
    b = 3
    l = 1
    level = 10
    crosswires = 1

    coordinates = get_coordinates(b, l, level, crosswires)
    grid_size = get_grid_size(b, level, crosswires)

    test = np.zeros((grid_size, grid_size))
    for coordinate in coordinates:
        test[coordinate[0]][coordinate[1]] = 1

    #plt.imshow(test)
    #plt.show()


if __name__ == '__main__':
    main()
