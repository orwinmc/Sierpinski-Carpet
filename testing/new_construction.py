import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_grid_size(b, level, crosswires):
    return (crosswires+1) * (b**level) + 1

def get_coordinates(b, l, level, crosswires):
    # Construct Base Object
    base = []
    for y in range(crosswires+2):
        for x in range(crosswires+2):
                base.append((y,x))



    # Eliminate Corners
    base.remove((0, 0))
    base.remove((crosswires+1, 0))
    base.remove((0, crosswires+1))
    base.remove((crosswires+1, crosswires+1))

    num_points = len(base)
    print(num_points)

    return base



        #new_coordinates.add((coordinate[0]+j*(get_grid_size(b, i, crosswires)-1) , coordinate[1]+k*(get_grid_size(b, i, crosswires)-1)))

        #coordinates = coordinates.union(new_coordinates)


def main():
    # Input Values
    b = 5
    l = 3
    level = 4
    crosswires = 1

    coordinates = get_coordinates(b, l, level, crosswires)
    grid_size = get_grid_size(b, level, crosswires)



if __name__ == '__main__':
    main()

    s = np.zeros((2,2))
    s = s+ (2,2)
    print(s)
