#pragma once
#include "waves.cu"
#include "grid1d.cu"
#include "grid2d.cu"

// Algorithms are devided between dimentions and uniform / non uniform velocity fields
namespace kernels {
    // 1D
    void init_step_uni_1D(SimSettings sets, Grid1D& g);
    __global__ void step_uni_1D(SimSettings, DeviceGrid g);

    // 2D
    typedef double (*init_v_2d)(double, double);
    void init_step_uni_2D(SimSettings sets, Grid2D& g, init_v_2d v);
    __global__ void step_uni_2D(SimSettings, DeviceGrid g);
};

// 1D
void kernels::init_step_uni_1D(SimSettings sets, Grid1D& g) {
    for (int i = 1; i < g.total_size-1; i++) {
        double x = i*sets.dx;
        double vel = 0.0;
        g.u[g.newi + i] = g.u[g.nowi + i] + sets.dt*vel + 0.5*sets.C*sets.C*(g.u[g.nowi + i+1] - 2*g.u[g.nowi + i] + g.u[g.nowi + i-1]) + 0.5*sets.dt*sets.dt*g.f[i];
    }
}

__global__ void kernels::step_uni_1D(SimSettings sets, DeviceGrid g) {
    int x = threadIdx.x;
    int y = blockIdx.x;

    int i = x; // + y * width;

    // If this isnt the boundary
    if (i != 0 && i != g.total_size-1) {
        if (g.types[i].id != CellID::DONT_UPDATE) {
            g.u[g.newi + i] = -g.u[g.oldi + i] + 2*g.u[g.nowi + i] + sets.C*sets.C*(g.u[g.nowi + i+1] - 2*g.u[g.nowi + i] + g.u[g.nowi + i-1]) + sets.dt*sets.dt*g.f[i];
        }

        // Boundary conditions are actally implemented just outside the boundary
        // They dont exclude updating the current cell
        if (g.types[i].id == CellID::BOUND_NEUMANN) {
            g.u[g.newi + i + g.types[i].out_delta] = g.u[g.newi + i - g.types[i].out_delta];
        }
    } else {
        // Handle possible periodic conditions of the cells are set to normal
        // if (g.types[i].id == CellID::NORMAL) {
        //     if (i == 0) {
        //         int l = g.total_size-1;
        //         int r = 1;
        //         g.u[g.newi] = -g.u[g.oldi] + 2*g.u[g.nowi] + sets.C*sets.C*(g.u[g.nowi + r] - 2*g.u[g.nowi + r] + g.u[g.nowi + l]) + sets.dt*sets.dt*g.f[i];
        //     } else {
        //         int l = g.total_size-2;
        //         int r = 0;
        //         g.u[g.newi] = -g.u[g.oldi] + 2*g.u[g.nowi] + sets.C*sets.C*(g.u[g.nowi + r] - 2*g.u[g.nowi + r] + g.u[g.nowi + l]) + sets.dt*sets.dt*g.f[i];
        //     }
        // }
    }
}

// 2D
void kernels::init_step_uni_2D(SimSettings sets, Grid2D& g, init_v_2d v) {
    int width = g.sizes[0];
    int height = g.sizes[1];

    for (int x = 1; x < width-1; x++) {
        for (int y = 1; y < height-1; y++) {
            double real_x = x*sets.dx;
            double real_y = y*sets.dx;
            double vel = v(real_x, real_y);
            
            // Index inside the grid
            int i = g.get_index(x, y);

            // right left down up
            int r = i+1;
            int l = i-1;
            int d = i-width;
            int up = i+width;
        
            g.u[g.newi + i] = g.u[g.nowi + i] + sets.dt*vel
                        + 0.5*sets.C*sets.C*(g.u[g.nowi+r] - 2*g.u[g.nowi+i] + g.u[g.nowi+l] + g.u[g.nowi+d] - 2*g.u[g.nowi+i] + g.u[g.nowi+up])
                        + 0.5*sets.dt*sets.dt*g.f[i];
        }
    }
}

__global__ void kernels::step_uni_2D(SimSettings sets, DeviceGrid g) {
    // X,Y in grid coordinates
    int x = threadIdx.x;
    int y = blockIdx.x;

    int width = g.sizes[0];
    int height = g.sizes[1];
    int i = x + y * width;

    // If this isnt the boundary
    if (x != 0 && x != width-1 && y != 0 && y != height-1) {
        int r = i+1;
        int l = i-1;
        int d = i-width;
        int up = i+width;

        if (g.types[i].id != CellID::DONT_UPDATE) {
            g.u[g.newi+i] = -g.u[g.oldi+i] + 2*g.u[g.nowi+i]
                        + sets.C*sets.C*(g.u[g.nowi+r] - 2*g.u[g.nowi+i] + g.u[g.nowi+l] + g.u[g.nowi+d] - 2*g.u[g.nowi+i] + g.u[g.nowi+up])
                        + sets.dt*sets.dt*g.f[i];
        }

        if (g.types[i].id == CellID::BOUND_NEUMANN) {
            g.u[g.newi + i + g.types[i].out_delta] = g.u[g.newi + i - g.types[i].out_delta];
        }
    }
}