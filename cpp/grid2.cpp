#include <iostream>
#include <fstream>
#include <cmath>
#include <ctime>
#include <vector>
#include <string>

#include <sys/resource.h> /* give extra space to stack `rlimit` */

#include <Eigen/Eigen> /* Sparse Matrices */

#include <omp.h> /* Parallel */

using namespace std;

int get_grid_size(int b, int level, int c) {
    return (c+1) * pow(b, level) + 1;
}

void get_grid_layout(int b, int l, int level, int c, long** grid_layout, int grid_size) {
    // Checks if the parameters given are valid
    if ((l!=0 && b%2 != l%2) || (b==l)) {
        cout << "Invalid Input!" << endl;
    } else {
        // Fill Matrix Initially
        for (int y = 0; y<grid_size; y++) {
            for (int x = 0; x<grid_size; x++) {
                if (x%(c+1) == 0 && y%(c+1) == 0) {
                    grid_layout[y][x] = -1;
                } else {
                    grid_layout[y][x] = -2;
                }
            }
        }

        // Fills in the holes in the grid
        for (int current_level = 1; current_level<=level; current_level++) {
            int current_grid_size = get_grid_size(b, current_level, c);
            int prev_grid_size = get_grid_size(b, current_level-1, c);
            int hole_size = l*(prev_grid_size-1)-1;
            //#pragma omp parallel collapse(4)
            for (int j = (b-l)/2*(prev_grid_size-1)+1; j<grid_size; j+=(current_grid_size-1)) {
                for (int i = (b-l)/2*(prev_grid_size-1)+1; i<grid_size; i+=(current_grid_size-1)) {
                    for (int y = 0; y<hole_size; y++) {
                        for (int x = 0; x<hole_size; x++) {
                            grid_layout[j+y][i+x] = -1;
                        }
                    }
                }
            }
        }
    }
}

void display_grid_layout(long** grid_layout, int grid_size) {
    for (int y = 0; y<grid_size; y++) {
        for (int x = 0; x<grid_size; x++) {
            if (grid_layout[y][x] != -1) {
                cout <<  "\u2588 ";
            } else {
                cout << "  ";
            }

        }
        cout << endl;
    }
}

struct point {
    int y;
    int x;
};

void print_point(point p) {
    cout << "{" << p.y << ", " << p.x << "}";
}

// -3 is the offset (starts -3, -4, -5 and converts to 0, 1, 2, ...)
// Needs to be confirmed
void get_adj_list(long** grid_layout, int grid_size, int crosswires, vector< vector<long>> & adjacency_list, vector<point> & coordinates) {
    long next_available_index = -3;

    // Left Boundary Assignment
    for (int y = 0; y<grid_size; y++) {
        if (grid_layout[y][0] == -2) {
            grid_layout[y][0] = next_available_index;
            coordinates.push_back({y, 0});
            next_available_index--;
        }
    }

    // Right Boundary Assignment
    for (int y = 0; y<grid_size; y++) {
        if (grid_layout[y][grid_size-1] == -2) {
            grid_layout[y][grid_size-1] = next_available_index;
            coordinates.push_back({y, grid_size-1});
            next_available_index--;
        }
    }

    // Beginning of Flood Fill Algorithm
    vector<point> stack;
    stack.push_back({1, 0});

    while (stack.size() > 0) {

        // Fetch Last element in stack
        point p = stack[stack.size()-1];
        stack.pop_back();
        if (grid_layout[p.y][p.x] < 0) {
            // Marks as visited
            grid_layout[p.y][p.x] = -grid_layout[p.y][p.x]-3;
            vector<long> adj_row;

            // Checks Left
            if (p.x > 0 && p.y % (crosswires+1) != 0 && grid_layout[p.y][p.x-1] != -1) {
                if (grid_layout[p.y][p.x-1] >= 0) {
                    adj_row.push_back(grid_layout[p.y][p.x-1]);
                } else {
                    if (grid_layout[p.y][p.x-1] == -2) {
                        grid_layout[p.y][p.x-1] = next_available_index;
                        coordinates.push_back({p.y, p.x-1});
                        next_available_index--;
                    }
                    adj_row.push_back(-grid_layout[p.y][p.x-1]-3);
                    stack.push_back({p.y, p.x-1});
                }
            }

            // Checks Right
            if (p.x < grid_size-1 && p.y % (crosswires+1) != 0 && grid_layout[p.y][p.x+1] != -1) {
                if (grid_layout[p.y][p.x+1] >= 0) {
                    adj_row.push_back(grid_layout[p.y][p.x+1]);
                } else {
                    if (grid_layout[p.y][p.x+1] == -2) {
                        grid_layout[p.y][p.x+1] = next_available_index;
                        coordinates.push_back({p.y, p.x+1});
                        next_available_index--;
                    }
                    adj_row.push_back(-grid_layout[p.y][p.x+1]-3);
                    stack.push_back({p.y, p.x+1});
                }
            }

            // Checks Up
            if (p.y > 0 && p.x % (crosswires+1) != 0 && grid_layout[p.y-1][p.x] != -1) {
                if (grid_layout[p.y-1][p.x] >= 0) {
                    adj_row.push_back(grid_layout[p.y-1][p.x]);
                } else {
                    if (grid_layout[p.y-1][p.x] == -2) {
                        grid_layout[p.y-1][p.x] = next_available_index;
                        coordinates.push_back({p.y-1, p.x});
                        next_available_index--;
                    }
                    adj_row.push_back(-grid_layout[p.y-1][p.x]-3);
                    stack.push_back({p.y-1, p.x});
                }
            }

            // Checks Down
            if (p.y < grid_size-1 && p.x % (crosswires+1) != 0 && grid_layout[p.y+1][p.x] != -1) {
                if (grid_layout[p.y+1][p.x] >= 0) {
                    adj_row.push_back(grid_layout[p.y+1][p.x]);
                } else {
                    if (grid_layout[p.y+1][p.x] == -2) {
                        grid_layout[p.y+1][p.x] = next_available_index;
                        coordinates.push_back({p.y+1, p.x});
                        next_available_index--;
                    }
                    adj_row.push_back(-grid_layout[p.y+1][p.x]-3);
                    stack.push_back({p.y+1, p.x});
                }
            }

            while (adjacency_list.size() <= grid_layout[p.y][p.x]) {
                vector<long> blank_row;
                adjacency_list.push_back(blank_row);
            }
            adjacency_list[grid_layout[p.y][p.x]] = adj_row;
        }
    }
}

void get_laplacian(Eigen::SparseMatrix<short> & laplacian, vector< vector<long>> & adjacency_list) {
    for (int i = 0; i<adjacency_list.size(); i++) {
        vector<long> row = adjacency_list[i];
        for (int j = 0; j<row.size(); j++) {
            laplacian.coeffRef(i, row[j]) = 1;
        }
        laplacian.coeffRef(i, i) = -row.size();
    }
}

void print_laplacian(Eigen::SparseMatrix<short> & laplacian) {
    cout << "Laplacian Matrix: " << endl;
    long num_coordinates = laplacian.cols();
    for (int i = 0; i<num_coordinates; i++) {
        for (int j = 0; j<num_coordinates; j++) {
            cout << laplacian.coeff(i, j) << " ";
        }
        cout << endl;
    }
}

void print_mtrx(Eigen::SparseMatrix<double> & mtrx) {
    cout << "Matrix: " << endl;
    long num_coordinates = mtrx.cols();
    for (int i = 0; i<mtrx.rows(); i++) {
        for (int j = 0; j<mtrx.cols(); j++) {
            cout << mtrx.coeff(i, j) << " ";
        }
        cout << endl;
    }
}

void find_potentials(Eigen::SparseMatrix<double> & potentials, vector< vector<long>> & adjacency_list) {
    /*struct rlimit lim;
    getrlimit(RLIMIT_STACK, &lim);
    cout << lim.rlim_cur << endl;*/

    long num_coordinates = adjacency_list.size();
    int num_computed_points = potentials.rows();
    int num_boundary_points = num_coordinates-num_computed_points;
    cout << "Total Boundary Points: " << num_boundary_points << endl;
    cout << "Computed Points: " << num_computed_points << endl;

    cout << "1) Initailizing Matrices ..." << endl;
    Eigen::SparseMatrix<double> r(num_computed_points, num_boundary_points);
    Eigen::SparseMatrix<double> a(num_computed_points, num_computed_points);

    cout << "2) Reserving Space ..." << endl;
    r.reserve(Eigen::VectorXi::Constant(num_boundary_points, 5));
    a.reserve(Eigen::VectorXi::Constant(num_computed_points, 5));

    cout << "3) Inserting Elements ..." << endl;
    for (int i = 0; i<adjacency_list.size(); i++) {
        if (i >= num_boundary_points) {
            vector<long> row = adjacency_list[i];

            // Adds elements not on diagonal
            for (int j = 0; j<row.size(); j++) {
                if (row[j] < num_boundary_points) {
                    r.insert(i-num_boundary_points, row[j]) = 1;
                } else {
                    a.insert(i-num_boundary_points, row[j]-num_boundary_points) = 1;
                }
            }

            // Add elements on the diagonal
            if (i < num_boundary_points) {
                r.insert(i-num_boundary_points, i) = -(double)row.size();
            } else {
                a.insert(i-num_boundary_points, i-num_boundary_points) = -(double)row.size();
            }
        }
    }
    //print_mtrx(a);
    //print_mtrx(r);

    cout << "4) Setting Boundary Conditions ..." << endl;
    Eigen::SparseMatrix<double> dirichlet(num_boundary_points, 1);
    for (int i = 0; i<num_boundary_points; i++) {
        if (i < num_boundary_points / 2) {
            dirichlet.coeffRef(i, 0) = 0;
        } else {
            dirichlet.coeffRef(i, 0) = 1;
        }
    }

    cout << "5) Solving ..." << endl;
    Eigen::SparseMatrix<double> b = -r*dirichlet;
    Eigen::SparseLU<Eigen::SparseMatrix<double> > solver;
    solver.analyzePattern(a);
    solver.factorize(a);
    //solver.setMaxIterations(1000);
    //solver.setTolerance(0.001);
    /*Eigen::Matrix<double, -1, 1> guess(num_computed_points);
    for (int i = 0; i<num_computed_points; i++) {
        guess[i] = 0.5;
    }*/
    //Eigen::SparseMatrix<double> x = lscg.solveWithGuess(b, guess);
    //Eigen::VectorXd x(num_computed_points);
    potentials = solver.solve(b);
    //print_mtrx(x);
}

// TO BE CLEANED UP
double max_difference(vector< vector<long>> & adjacency_list, Eigen::SparseMatrix<double> & potentials) {
    long num_coordinates = adjacency_list.size();
    int num_computed_points = potentials.rows();
    int num_boundary_points = num_coordinates-num_computed_points;

    double max_diff = 0;

    for (int i = 0; i<adjacency_list.size(); i++) {
        // figure out potential
        double potential_a = 1;
        if (i < num_boundary_points/2) {
            potential_a = 0;
        } else if (i >= num_boundary_points) {
            potential_a = potentials.coeff(i-num_boundary_points, 0);
        }

        vector<long> row = adjacency_list[i];
        for (int j = 0; j<row.size(); j++) {
            long index = row[j];

            // figure out potential
            double potential_b = 1;
            if (index < num_boundary_points/2) {
                potential_b = 0;
            } else if (index >= num_boundary_points) {
                potential_b = potentials.coeff(index-num_boundary_points, 0);
            }

            double diff = abs(potential_a-potential_b);

            if (diff-max_diff > 0.0000001) {
                //cout << diff << endl;
                max_diff = diff;
            }
        }
    }

    return max_diff;
}
// THIS SECTION ABOVE

int harmonic_function(int b, int l, int level, int crosswires, string filename) {
    /*struct rlimit lim;
    getrlimit(RLIMIT_STACK, &lim);
    cout << lim.rlim_cur << endl;

    const rlimit stack_size = {16*1024*1024, 16*1024*1024};
    if (setrlimit(RLIMIT_STACK, &stack_size) == -1) {
        return 1;
    }*/
    // Parameters
    /*const int b = 3;
    const int l = 1;
    const int level = 1;
    const int crosswires = 1;*/

    // Making Grid Layout
    cout << "Making Grid Layout ..." << endl;
    int grid_size = get_grid_size(b, level, crosswires);
    cout << "Grid Size: " << grid_size << endl;
    long ** grid_layout = new long*[grid_size];
    for (int i = 0; i<grid_size; i++) {
        grid_layout[i] = new long[grid_size];
    }
    get_grid_layout(b, l, level, crosswires, grid_layout, grid_size);

    // View Grid Layout in Terminal
    //display_grid_layout(grid_layout, grid_size);

    // Adjacency List Calculation
    cout << "Making Adjacency List ..." << endl;
    vector< vector<long>> adjacency_list;
    vector<point> coordinates;
    get_adj_list(grid_layout, grid_size, crosswires, adjacency_list, coordinates);

    // FREE GRID LAYOUT
    for(int i =0 ; i<grid_size; i++) {
        delete[] grid_layout[i];
    }
    delete[] grid_layout;

    // Computes Laplacian
    long num_coordinates = adjacency_list.size();
    cout << "num_coordinates: " << num_coordinates << endl;
    //Eigen::SparseMatrix<short> laplacian(num_coordinates, num_coordinates);
    //get_laplacian(laplacian, adjacency_list);

    // Computes Potentials
    cout << "Compute Potentials ..." << endl;
    int num_boundary_points = 2*crosswires*pow(b,level);
    int num_computed_points = num_coordinates-num_boundary_points;
    Eigen::SparseMatrix<double> potentials(num_computed_points, 1);
    find_potentials(potentials, adjacency_list);

    // To be fixed up
    double max_diff = max_difference(adjacency_list, potentials);
    cout << max_diff << endl;



    // Output to file
    /*ofstream fout(filename);

    time_t now = time(0);
    string dt = ctime(&now);
    fout << "Produced by \"The Resistance\", " << dt;
    fout << "---------------------------------------------------------" << endl;
    fout << "Parameters: b = " << b << ", l = " << l << ", level = " << level << ", crosswires = " << crosswires << endl;
    fout << "---------------------------------------------------------" << endl;
    fout << "y\tx\tpotential" << endl;
    for (int i = 0; i<num_coordinates; i++) {
        if (i < num_boundary_points) {
            if (i < num_boundary_points/2) {
                fout << coordinates[i].y << "\t" << coordinates[i].x << "\t" << 0 << endl;
            } else {
                fout << coordinates[i].y << "\t" << coordinates[i].x << "\t" << 1 << endl;
            }
        } else {
            fout << coordinates[i].y << "\t" << coordinates[i].x << "\t" << potentials.coeff(i-num_boundary_points, 0) << endl;
        }
    }
    fout.close();*/
    return 0;
}

int main() {
    int b = 3;
    int l = 1;
    int level = 8;
    int crosswires = 1;
    /*for (int level = 0; level<8; level++) {
        cout << "----------------------\n";
        string filename = "../data/plus/plusdata_"+to_string(b)+"_"+to_string(l)+"_"+to_string(crosswires)+"_level"+to_string(level)+".dat";
        //cout << filename;
        harmonic_function(b, l, level, crosswires, filename);
    }*/

    harmonic_function(b, l, level, crosswires, "asdf");

}

// Test
// test again
