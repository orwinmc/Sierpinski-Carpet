#include <iostream>
#include <cmath>
#include <omp.h>

using namespace std;

int get_grid_size(int b, int level, int c) {
    return (c+1) * pow(b, level) + 1;
}

void get_grid_layout(int b, int l, int level, int c, bool** grid_layout, int grid_size) {
    // Checks if the parameters given are valid
    if ((l!=0 && b%2 != l%2) || (b==l)) {
        cout << "Invalid Input!" << endl;
    } else {
        // Fill Matrix Initially
        for (int y = 0; y<grid_size; y++) {
            for (int x = 0; x<grid_size; x++) {
                if (x%(c+1) == 0 && y%(c+1) == 0) {
                    grid_layout[y][x] = false;
                } else {
                    grid_layout[y][x] = true;
                }
            }
        }

        // Fills in the holes in the grid
        for (int current_level = 1; current_level<=level; current_level++) {
            int current_grid_size = get_grid_size(b, current_level, c);
            int prev_grid_size = get_grid_size(b, current_level-1, c);
            int hole_size = l*(prev_grid_size-1)-1;
            for (int j = (b-l)/2*(prev_grid_size-1)+1; j<grid_size; j+=(current_grid_size-1)) {
                for (int i = (b-l)/2*(prev_grid_size-1)+1; i<grid_size; i+=(current_grid_size-1)) {
                    for (int y = 0; y<hole_size; y++) {
                        for (int x = 0; x<hole_size; x++) {
                            grid_layout[j+y][i+x] = false;
                        }
                    }
                }
            }
        }
    }
}

int main() {
    int b = 3;
    int l = 1;
    int level = 9;
    int c = 1;

    // Making Grid Layout
    int grid_size = get_grid_size(b, level, c);
    cout << grid_size << endl;
    bool ** grid_layout = new bool*[grid_size];
    for (int i = 0; i<grid_size; i++) {
        grid_layout[i] = new bool[grid_size];
    }
    get_grid_layout(b, l, level, c, grid_layout, grid_size);

    // View
    /*for (int y = 0; y<grid_size; y++) {
        for (int x = 0; x<grid_size; x++) {
            cout << grid_layout[y][x] << " ";
        }
        cout << endl;
    }*/

    #pragma omp parallel
    for (int i = 0; i<9000000; i++) {
        //cout << i << endl;
    }
    cout << "DONE" << endl;

    // NEED TO FREE GRID LAYOUT
    return 0;
}



/*def get_grid_layout(b, l, level, c):

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
        for i in range((b-l)/2*(prev_vertices_current-1)+1, num_vertices, vertices_current-1):
            for j in range((b-l)/2*(prev_vertices_current-1)+1, num_vertices, vertices_current-1):
                sc[i:i+hole_size, j:j+hole_size] = 0

    return sc*/
