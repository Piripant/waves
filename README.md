# Waves
A complete rewrite of the older waves simulation program, supports 1D, 2D, 3D wave solution with CUDA hardware acceleration.

## The Grid
Apart from the information about the wave passing each cell on the grid, there are other additional tags associated with each point:
```c++
enum class CellID {
    NORMAL = 0,
    DONT_UPDATE = 1,
    BOUND_NEUMANN = 2,
    BOUND_OPEN = 3
};

class CellType {
    public:
        CellID id;
        int out_delta;
};
```
This structure is used to build boundary conditions, BOTH at the edges of the grid, and in the middle.

Different boundaries are built in different ways, because of the underlying differences in the implementation and mathematical definition of each.

### Dirichlet boundaries
Dirichlet boundaries are defined as points on which the grid is stationary: $u_i^n = 0$.
To implement this kind of behaviour you need to set the interested cell to DONT_UPDATE, and the value of the cell on the desired block will stay constant.

### Neumann boundaries
Neumann boudaries are defined as points on which the derivative of wave field is stationary: $\frac{\partial u^n}{\partial x} = 0$.

This condiotions are trickier and require to set two cells to different types, take this 1D example.
```
XXOXX -> XNDNX
```
We want to be and obstacle onto which waves reflect off of. What we need to set is pictured after the right arrow, where N stands for a cell marked as BOUND_NEUMANN and D as DONT_UPDATE.

We notice that cells marked BOUND_NEUMANN aren't actually on the wall, but right besides it. This is because neumann boundary conditions are implemented just besides the wall, as setting the derivative to zero is done with a central discretization of the function.

### Periodic boundaries
Periodic boundaries are implemented by default along the boundaries of our grid, but can be overwritten by settings special boundaries like the ones described for special or dirichlet boundaries