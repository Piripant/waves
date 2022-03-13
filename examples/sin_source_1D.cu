#include "../src/grid1d.cu"
#include "../src/kernels.cu"
#include <math.h>

__global__ void sin_func(DeviceGrid grid, int pos, double t) {
    if (t < 1.0) {
        grid.u[grid.nowi + pos] = sin(2*M_PI*t);
    } else {
        grid.u[grid.nowi + pos] = 0;
    }
}

int main() {
    double length = 2.0;
    double tmax = 5.0;

    SimSettings sets = SimSettings::uniform_v(0.01, 0.9, 0.5);
    int width = length / sets.dx;
    if (width > 1024) {
        printf("Width too big\n");
        return -1;
    }
    
    Grid1D grid(width);
    grid.set_type(width/2, CellID::DONT_UPDATE);
    grid.set_type(1, CellID::BOUND_NEUMANN);
    grid.set_type(width-2, CellID::BOUND_NEUMANN);
    grid.gen_bounds();
    
    kernels::init_step_uni_1D(sets, grid);
    grid.oldi = (grid.oldi + grid.total_size) % (3*grid.total_size);
    grid.nowi = (grid.nowi + grid.total_size) % (3*grid.total_size);
    grid.newi = (grid.newi + grid.total_size) % (3*grid.total_size);

    DeviceGrid d_grid;
    grid.to_device(&d_grid);

    FILE* wavef;
    wavef = fopen("out.dat", "w");

    // Next steps
    for (int t = 1; t < tmax / sets.dt; t++) {
        // Update the wave
        sin_func<<<1,1>>>(d_grid, width/2, (double)t * sets.dt);
        kernels::step_uni_1D<<<1, d_grid.total_size>>>(sets, d_grid);
        cudaDeviceSynchronize();

        // Copy grid back to RAM, print and compute error
        fprintf(wavef, "%lf\n", t*sets.dt);
        grid.copy_present(&d_grid);
        printf("%.3lf, %.3lf\n", grid.u[width/2], sin((double)t * sets.dt));
        
        for (int i = 0; i < grid.total_size; i++) {
            fprintf(wavef, "%lf %lf\n", i*sets.dx, grid.u[grid.nowi + i]);
        }
        fprintf(wavef, "\n\n");

        // Progress
        d_grid.oldi = (d_grid.oldi + d_grid.total_size) % (3*d_grid.total_size);
        d_grid.nowi = (d_grid.nowi + d_grid.total_size) % (3*d_grid.total_size);
        d_grid.newi = (d_grid.newi + d_grid.total_size) % (3*d_grid.total_size);
    }

    fclose(wavef);
    d_grid.destroy();
}