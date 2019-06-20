# General Imports
from __future__ import print_function
import sys
import argparse

# Math Imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.sparse.linalg as la

def get_grid_size(b, level):
    return 2 * b**level + 1

def get_grid_layout(b, l, level):
    print('Generating Grid Layout ...')

    if (l!=0 and b%2 != l%2) or b==l:
        print("Invalid Input!")
        sys.exit(1)
    else:
        grid_size = get_grid_size(b, level)
        layout = np.empty((grid_size, grid_size), dtype=object)

        # Removes non-diagonal holes formed by each x
        for y, row in enumerate(layout):
            for x, val in enumerate(row):
                if x%2 != y%2:
                    layout[y, x] = -1
                else:
                    layout[y, x] = -2

        # Iterates over each size hole removing it from the layout
        for current_level in range(1, level+1):
            current_grid_size = get_grid_size(b, current_level)
            prev_grid_size = get_grid_size(b, current_level-1)
            hole_size = l*(prev_grid_size-1)-1
            for i in range((b-l)//2*(prev_grid_size-1)+1, grid_size, current_grid_size-1):
                for j in range((b-l)//2*(prev_grid_size-1)+1, grid_size, current_grid_size-1):
                    layout[i:i+hole_size, j:j+hole_size] = -1

    return layout

def display_grid_layout(layout, display_type='grid'):
    ''' For viewing the fractal (SC) layout. Two options are available
    ("terminal" and "matplotlib") '''

    if display_type == 'matplotlib':
        plt.imshow(layout.astype(int))
        plt.show()
    elif display_type == 'terminal':
        for y, row in enumerate(layout):
            for x, val in enumerate(row):
                if val == -2:
                    print(u"\u2588 ", end='')
                else:
                    print("  ", end='')
            print()

def main():
    # Make printing a bit nicer for visualizing
    np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

    # Algorithm Parameters (Type -h for usage)
    parser = argparse.ArgumentParser(description='Generates the x Graph Approximations for the Sierpinski Carpet')
    parser.add_argument('-b', default=3, type=int, help='The number of sections to divide the carpet into')
    parser.add_argument('-l', default=1, type=int, help='The number of sections to remove from the carpet center')
    parser.add_argument('-a', '--level', type=int, default=3, help='Number of pre-carpet contraction iterations')
    args = parser.parse_args()

    # Begin Computation of Harmonic Function
    print('Generating x Graph Approximation for b=%d, l=%d, level=%d ...' % (args.b, args.l, args.level))
    grid_size = get_grid_size(args.b, args.level)
    layout = get_grid_layout(args.b, args.l, args.level)
    display_grid_layout(layout, display_type='matplotlib')

if __name__ == '__main__':
    main()
