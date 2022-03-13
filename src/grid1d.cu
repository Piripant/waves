#pragma once
#include "waves.cu"
#include <stdexcept>

class Grid1D : public Grid {
    public:
        Grid1D(int width) : Grid(std::vector<int>{width}) {
            // Set ghost cells for the grid
            this->types[0].id = CellID::DONT_UPDATE;
            this->types[width-1].id = CellID::DONT_UPDATE;
        };
        bool in_bounds(int index) {
            if (index < 0 || index >= total_size) {
                return false;
            } else {
                return true;
            }
        };
        void set_type(int index, CellID id);
        void gen_bounds();
};

void Grid1D::set_type(int index, CellID id) {
    if (!in_bounds(index)) {
        throw std::out_of_range("Cell is out of bounds");
    }

    if (index == 0 || index == total_size-1) {
        // and throw an exception because we are not allowed to modify the ghost cells
        throw std::invalid_argument("Setting type of cell 0 or length-1 is forbidden");
    }

    this->types[index].id = id;
}

void Grid1D::gen_bounds() {
    for (int i = 1; i < total_size; i++) {
        CellID id = types[i].id;
        // We have to find which neighbour is outside of the boundary
        if (id == CellID::BOUND_NEUMANN || id == CellID::BOUND_OPEN) {
            if (this->types[i+1].id == CellID::DONT_UPDATE) {
                this->types[i].out_delta = 1;
            } else if (this->types[i-1].id == CellID::DONT_UPDATE) {
                this->types[i].out_delta = -1;
            } else {
                throw std::invalid_argument("A boundary cell did not neighbour a DONT_UPDATE cell");
            }
        }
    }
}