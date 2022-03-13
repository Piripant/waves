#include "../src/grid1d.cu"
#include "../src/kernels.cu"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

const double length = 1.0;

double init_pos(double x) {
    return sin(2*M_PI/length * x);
}

double exact(double x, double t) {
    return cos(2*M_PI/length * t)*sin(2*M_PI/length * x);
}

// Simulate for t_max seconds
double solve_wave(SimSettings sets, Grid1D& grid, double t_max) {
    // Set initial conditions for the grid
    for (int i = 0; i < grid.total_size; i++) {
        grid.u[grid.nowi + i] = init_pos(i*sets.dx);
    }

    // Initial step and update grids
    kernels::init_step_uni_1D(sets, grid);
    grid.advance();

    // Generate device image of the grid
    DeviceGrid d_grid;
    grid.to_device(&d_grid);

    FILE* wavef;
    wavef = fopen("out.dat", "w");

    // Next steps
    double total_error = 0.0;
    for (int t = 1; t < t_max / sets.dt; t++) {
        // Update the wave
        kernels::step_uni_1D<<<1, d_grid.total_size>>>(sets, d_grid);
        cudaDeviceSynchronize();

        // Copy grid back to RAM, print and compute error
        fprintf(wavef, "%lf\n", t*sets.dt);
        grid.copy_present(&d_grid);
        
        for (int i = 0; i < grid.total_size; i++) {
            fprintf(wavef, "%lf %lf\n", i*sets.dx, grid.u[grid.nowi + i]);
            total_error = fmax(fabs(grid.u[grid.nowi + i] - exact(i*sets.dx, t*sets.dt)), total_error);
        }
        fprintf(wavef, "\n\n");

        // Progress
        d_grid.advance();
    }

    fclose(wavef);
    d_grid.destroy();

    return total_error;
}

int main() {
    SimSettings sets;
    double v = 1.0;

    sets.dt = 0.1;
    sets.C = 0.9;

    double old_dt;
    double old_err;

    for (int n = 0; n < 6; n++) {
        sets.dx = sets.dt*v/sets.C;
        int width = length / sets.dx + 1;
        Grid1D grid(width);

        grid.set_uniform_speed(v);
        
        if (width > 1024) {
            printf("Maximum width has been reached\n");
            return -1;
        }

        double error = solve_wave(sets, grid, 1.0);
        
        if (n == 0) {
            printf("dx: %lf dt: %lf error: %lf\n", sets.dx, sets.dt, error);
        } else {
            double r = log(error/old_err)/log(sets.dt/old_dt);
            printf("dx: %lf dt: %lf error: %lf r: %lf\n", sets.dx, sets.dt, error, r);
        }

        old_dt = sets.dt;
        old_err = error;
        sets.dt /= 2;
    }
}