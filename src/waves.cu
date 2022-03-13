#pragma once
#include <vector>
#include <stdio.h>

using namespace std;

// Settings for the simulation, which include the dx and dt
class SimSettings {
    public:
        double dt;
        double dx;
        double C;

        static SimSettings uniform_v(double dt, double C, double v) {
            SimSettings sets;
            sets.dt = dt;
            sets.C = C;
            sets.dx = sets.dt*v/sets.C;
            return sets;
        }
};

enum class CellID {
    NORMAL = 0,
    DONT_UPDATE = 1, // This can be used for dirichlet boundaries too
    BOUND_NEUMANN = 2,
    BOUND_OPEN = 3
};

class CellType {
    public:
        CellID id;
        // Only some need this
        // Relative index distance to neighbouring points outside boundary
        // index + out_delta = outside_point_index
        int out_delta;
};

// Device copy of all the data needed for simulation
// Any dynamic f non homogeneos function should be updated inside the kernel itself.
class DeviceGrid {
    public:
        double *v;
        double *u;
        double *f;
        int *sizes;

        CellType *types;
        bool generated_types;

        int total_size;
        int oldi;
        int nowi;
        int newi;

        void destroy() {
            cudaFree(this->v);
            cudaFree(this->u);
            cudaFree(this->f);
            cudaFree(this->sizes);
        }

        void advance() {
            oldi = (oldi + total_size) % (3*total_size);
            nowi = (nowi + total_size) % (3*total_size);
            newi = (newi + total_size) % (3*total_size);
        }
};

// Main host structure, which holds all the data on the cpu
class Grid {
    public:
        double *v;
        double *u;
        double *f;
        CellType *types;

        int nowi;
        int newi;
        int oldi;

        int total_size;
        vector<int> sizes;

        Grid(vector<int> sizes);
        ~Grid();

        // Generate the best simulation settings for the current use-case
        SimSettings get_settings(double dt, double C);
        void set_uniform_speed(double v);

        void to_device(DeviceGrid *d_grid);
        void copy_present(DeviceGrid *d_grid);

        void advance();
};

Grid::Grid(vector<int> sizes) {
    if (sizes.size() > 3) {
        printf("Initiated with more than max dimension");
        return;
    }
    this->sizes = sizes;
    this->total_size = 1;
    for (int i = 0; i < sizes.size(); i++) {
        this->total_size *= sizes[i];
    }

    this->u = (double*)calloc(this->total_size * 3, sizeof(double));
    this->v = (double*)calloc(this->total_size, sizeof(double));
    this->f = (double*)calloc(this->total_size, sizeof(double));
    this->types = (CellType*)calloc(this->total_size, sizeof(CellType));
    
    this->oldi = 0;
    this->nowi = this->total_size;
    this->newi = this->total_size*2;
}

Grid::~Grid() {
    free(this->v);
    free(this->u);
    free(this->f);
}

void Grid::set_uniform_speed(double v) {
    for (int i = 0; i < total_size; i++) {
        this->v[i] = v;
    }
}

void Grid::to_device(DeviceGrid *d_grid) {
    cudaMalloc((void **) &d_grid->u, this->total_size * 3 * sizeof(double));
    cudaMalloc((void **) &d_grid->f, this->total_size * sizeof(double));
    cudaMalloc((void **) &d_grid->v, this->total_size * sizeof(double));
    cudaMalloc((void **) &d_grid->types, this->total_size * sizeof(CellType));
    cudaMalloc((void **) &d_grid->sizes, this->sizes.size() * sizeof(int));

    cudaMemcpy(d_grid->u, this->u, this->total_size * 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grid->f, this->f, this->total_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grid->v, this->v, this->total_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grid->types, this->types, this->total_size * sizeof(CellType), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grid->sizes, this->sizes.data(), sizeof(int) * this->sizes.size(), cudaMemcpyHostToDevice);

    d_grid->total_size = total_size;
    d_grid->oldi = this->oldi;
    d_grid->nowi = this->nowi;
    d_grid->newi = this->newi;
}

void Grid::copy_present(DeviceGrid *d_grid) {
    this->oldi = d_grid->oldi;
    this->nowi = d_grid->nowi;
    this->newi = d_grid->newi;

    cudaMemcpy(&this->u[0], &d_grid->u[0], this->total_size * sizeof(double) * 3, cudaMemcpyDeviceToHost);
}

void Grid::advance() {
    oldi = (oldi + total_size) % (3*total_size);
    nowi = (nowi + total_size) % (3*total_size);
    newi = (newi + total_size) % (3*total_size);
}