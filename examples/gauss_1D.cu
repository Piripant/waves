#include "../src/kernels.cu"
#include "../src/grid1d.cu"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

const double length = 2.0;

double init_pos(double x, double center, double stddev) {
    return exp(-(x-center)*(x-center)/(2*stddev*stddev))/(stddev*sqrt(2*3.14159));
}

// Simulate for t_max seconds
void solve_wave(SimSettings sets, Grid1D& grid, double t_max) {
    // Initial step and update grids
    kernels::init_step_uni_1D(sets, grid);
    grid.oldi = (grid.oldi + grid.total_size) % (3*grid.total_size);
    grid.nowi = (grid.nowi + grid.total_size) % (3*grid.total_size);
    grid.newi = (grid.newi + grid.total_size) % (3*grid.total_size);

    // Generate device image of the grid
    DeviceGrid d_grid;
    grid.to_device(&d_grid);

    FILE* wavef;
    wavef = fopen("out.dat", "w");

    // Next steps
    for (int t = 1; t < t_max / sets.dt; t++) {
        // Update the wave
        kernels::step_uni_1D<<<1, d_grid.total_size>>>(sets, d_grid);
        cudaDeviceSynchronize();

        // Copy grid back to RAM, print and compute error
        fprintf(wavef, "%lf\n", t*sets.dt);
        grid.copy_present(&d_grid);
        
        for (int i = 0; i < grid.total_size; i++) {
            fprintf(wavef, "%lf %lf\n", i*sets.dx, grid.u[grid.nowi + i]);
        }

        // Progress
        d_grid.oldi = (d_grid.oldi + d_grid.total_size) % (3*d_grid.total_size);
        d_grid.nowi = (d_grid.nowi + d_grid.total_size) % (3*d_grid.total_size);
        d_grid.newi = (d_grid.newi + d_grid.total_size) % (3*d_grid.total_size);
    }

    fclose(wavef);
    d_grid.destroy();
}

int main() {
    SimSettings sets;
    double v = 1.0;
    double tmax = 1.0;

    sets.dt = 0.01;
    sets.C = 0.9;

    sets.dx = sets.dt*v/sets.C;
    int width = length / sets.dx + 1;
    if (width > 1024) {
        printf("Maximum width has been reached\n");
        return -1;
    }
    Grid1D grid(width);

    grid.set_uniform_speed(v);
    grid.set_type(1, CellID::BOUND_NEUMANN);
    grid.set_type(grid.total_size - 2, CellID::BOUND_NEUMANN);

    int block_size = 5;
    for (int i = 0; i < block_size; i++) {
        grid.set_type(grid.total_size / 2 + i, CellID::DONT_UPDATE);
        grid.set_type(grid.total_size / 2 - i, CellID::DONT_UPDATE);
    }
    grid.set_type(grid.total_size / 2 + block_size, CellID::BOUND_NEUMANN);
    grid.set_type(grid.total_size / 2 - block_size, CellID::BOUND_NEUMANN);

    grid.gen_bounds();

    // Set initial conditions for the grid
    for (int i = 0; i < grid.total_size; i++) {
        if (grid.types[i].id != CellID::DONT_UPDATE) {
            grid.u[grid.nowi + i] = init_pos(i*sets.dx, 0.5, 0.05) + init_pos(i*sets.dx, 1.5, 0.05);
        }
    }

    solve_wave(sets, grid, tmax);

    printf("total_steps: %d\n", (int)(1.0 / sets.dt));
}