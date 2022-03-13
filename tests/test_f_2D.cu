#include "../src/waves.cu"
#include "../src/grid2d.cu"
#include "../src/kernels.cu"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

const double length = 1.0;
const double v = 1.0;

double exact(double x, double y, double t) {
    return x * (length - x) * y * (length - y)*(1 + 0.5*t);    
}

double init_pos(double x, double y) {
    return exact(x, y, 0);
}

double init_vel(double x, double y) {
    return 0.5*exact(x, y, 0);
}

__host__ __device__ double f(double x, double y, double t) {
    return 2*v*v*(1 + 0.5*t)*(y*(length - y) + x*(length - x));
}

__global__ void f_arr(SimSettings sets, DeviceGrid g, double t) {
    int x = threadIdx.x;
    int y = blockIdx.x;

    int i = x + y * g.sizes[0];
    
    g.f[i] = f(x*sets.dx, y*sets.dx, t);
}

// Simulate for t_max seconds
double solve_wave(SimSettings sets, Grid2D& grid, double t_max) {
    int width = grid.sizes[0];
    int height = grid.sizes[1];
    
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            grid.u[grid.nowi+grid.get_index(x, y)] = init_pos(x*sets.dx, y*sets.dx);
            grid.f[grid.get_index(x, y)] = f(x*sets.dx, y*sets.dx, 0);
        }
    }

    kernels::init_step_uni_2D(sets, grid, init_vel);
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
        f_arr<<<height, width>>>(sets, d_grid, t*sets.dt);
        cudaDeviceSynchronize();
        kernels::step_uni_2D<<<height, width>>>(sets, d_grid);
        cudaDeviceSynchronize();

        // Copy grid back to RAM, print and compute error
        fprintf(wavef, "%lf\n", t*sets.dt);
        grid.copy_present(&d_grid);

        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                fprintf(wavef, "%lf %lf %lf\n", x*sets.dx, y*sets.dx, grid.u[grid.nowi + grid.get_index(x, y)]);
                double diff = grid.u[grid.nowi + grid.get_index(x, y)] - exact(x*sets.dx, y*sets.dx, t*sets.dt);
                total_error = fmax(fabs(diff), total_error);
            }
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

    sets.dt = 0.1;
    sets.C = 0.7;

    double old_dt;
    double old_err;

    for (int n = 0; n < 6; n++) {
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