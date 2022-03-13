#include "src/waves.cu"

__global__ void erase(DeviceGrid g) {
    int x = threadIdx.x;
    int y = blockIdx.x;

    int i = x;
    
    g.u[g.nowi + i] = 7;
    printf("d %d %lf\n", i, g.u[g.nowi + i]);
}

int main () {
    vector<int> sizes{10};    
    Grid grid(sizes);
    for (int i = 0; i < grid.total_size; i++) {
        grid.u[grid.nowi + i] = 2;
    }

    // Generate device image of the grid
    DeviceGrid d_grid;
    grid.to_device(&d_grid);
    erase<<<1, d_grid.total_size>>>(d_grid);
    grid.copy_present(&d_grid);
    
    for (int i = 0; i < grid.total_size; i++) {
        printf("h %d %lf\n", i, grid.u[grid.nowi + i]);
    }
}