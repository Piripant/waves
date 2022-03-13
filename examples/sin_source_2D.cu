#include "../src/waves.cu"
#include "../src/grid2d.cu"
#include "../src/kernels.cu"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

const double length = 2.0;
const double v = 1.0;

__global__ void sin_func(DeviceGrid grid, int pos, double t) {
    grid.u[grid.nowi + pos] = sin(2*M_PI*t*2);
}

double init_vel(double x, double y) {
    return 0.0;
}

// Simulate for t_max seconds
void solve_wave(SimSettings sets, Grid2D& grid, double t_max) {
    int width = grid.sizes[0];
    int height = grid.sizes[1];
    
    kernels::init_step_uni_2D(sets, grid, init_vel);
    grid.advance();

    // Generate device image of the grid
    DeviceGrid d_grid;
    grid.to_device(&d_grid);

    FILE* wavef;
    wavef = fopen("out.dat", "w");

    // Next steps
    for (int t = 1; t < t_max / sets.dt; t++) {
        // Update the wave
        sin_func<<<1,1>>>(d_grid, grid.get_index(0.25/sets.dx, length*0.5/sets.dx), (double)t * sets.dt);
        cudaDeviceSynchronize();
        kernels::step_uni_2D<<<height, width>>>(sets, d_grid);
        cudaDeviceSynchronize();
        // Progress
        d_grid.advance();

        // Copy grid back to RAM, print and compute error
        fprintf(wavef, "%lf\n", t*sets.dt);
        grid.copy_present(&d_grid);

        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                fprintf(wavef, "%lf %lf %lf %lf %lf\n", x*sets.dx, y*sets.dx, grid.u[grid.nowi + grid.get_index(x, y)]);
            }
        }
        fprintf(wavef, "\n\n");
    }

    fclose(wavef);
    d_grid.destroy();
}

int main() {
    SimSettings sets;

    sets.dt = 0.01;
    sets.C = 0.7;

    sets.dx = sets.dt*v/sets.C;
    int width = length / sets.dx + 1;
    if ((width-1)*sets.dx < length) {
        width++;
    }
    int height = width;

    if (width > 1024) {
        printf("Maximum width has been reached\n");
        return -1;
    }

    Grid2D grid(width, height);
    int slith_size = 10;
    for (int y = 2; y <= height/2-slith_size; y++) {
        grid.set_type(width/4-1, y, CellID::BOUND_NEUMANN);
        grid.set_type(width/4, y, CellID::DONT_UPDATE);
        grid.set_type(width/4+1, y, CellID::BOUND_NEUMANN);
    }

    for (int y = height/2+slith_size; y < height-2; y++) {
        grid.set_type(width/4-1, y, CellID::BOUND_NEUMANN);
        grid.set_type(width/4, y, CellID::DONT_UPDATE);
        grid.set_type(width/4+1, y, CellID::BOUND_NEUMANN);
    }
    
    grid.gen_bounds();

    solve_wave(sets, grid, 5.0);
}