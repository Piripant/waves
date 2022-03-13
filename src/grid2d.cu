#pragma once
#include "waves.cu"
#include <stdexcept>

class Grid2D : public Grid {
    public:
        Grid2D(int width, int height) : Grid(std::vector<int>{width, height}) {
            // Set ghost cells for the grid
            for (int x = 0; x < width; x++) {
                this->types[get_index(x, 0)].id = CellID::DONT_UPDATE;
                this->types[get_index(x, height-1)].id = CellID::DONT_UPDATE;
            }
            for (int y = 0; y < height; y++) {
                this->types[get_index(0, y)].id = CellID::DONT_UPDATE;
                this->types[get_index(width-1, y)].id = CellID::DONT_UPDATE;
            }
        };

        bool in_bounds(int x, int y) {
            if (x < 0 || x >= sizes[0] || y < 0 || y >= sizes[1]) {
                return false;
            } else {
                return true;
            }
        };

        int get_index(int x, int y) {
            if (!in_bounds(x, y)) {
                throw std::out_of_range("Cell is out of bounds");
            } else {
                return x + y * sizes[0];
            }
        };

        void set_type(int x, int y, CellID id);
        void gen_bounds();
};

void Grid2D::set_type(int x, int y, CellID id) {
    if (!in_bounds(x, y)) {
        throw std::out_of_range("Cell is out of bounds");
    }

    if (x == 0 || y == 0 || x == sizes[0]-1 || y == sizes[1]-1) {
        // and throw an exception because we are not allowed to modify the ghost cells
        throw std::invalid_argument("Setting type of grid boundary cells is forbidden");
    }

    this->types[get_index(x, y)].id = id;
}

void Grid2D::gen_bounds() {
    for (int x = 1; x < sizes[0]-1; x++) {
        for (int y = 1; y < sizes[1]-1; y++) {
            int i = get_index(x, y);
            CellID id = types[i].id;

            if (id == CellID::BOUND_NEUMANN || id == CellID::BOUND_OPEN) {
                // Find which neighbour is the DONT_UPDATE cell
                // We also make sure that there is one DONT_UPDATE neighbour cell
                // As our algorithm defines behaviour only with one of these neighbouring cells
                bool found_neighbor = false;
                int dir[4][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
                for (int j = 0; j < 4; j++) {
                    int dx = dir[j][0];
                    int dy = dir[j][1];
                    int ni = get_index(x + dx, y + dy);
                    
                    if (this->types[ni].id == CellID::DONT_UPDATE) {
                        if (found_neighbor) {
                            throw std::invalid_argument("A boundary cell has more than one DONT_UPDATE neighbour cell");
                        }
                        // out_delta is defined as the difference between these indexes, such that
                        // index + types[index].out_delta = neighbour_index
                        this->types[i].out_delta = ni - i;
                        found_neighbor = true;
                    }
                }

                if (!found_neighbor) {
                    throw std::invalid_argument("A boundary cell did not neighbour a DONT_UPDATE cell");
                }
            }
        }
    }
}