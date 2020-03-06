from fenics import *
import matplotlib as pltlib
pltlib.use('Agg')  # To be able to plot
import numpy as np
from ufl.operators import nabla_div
import matplotlib.pyplot as plt
from mshr import *

### Program for multiple boundary conditions - two Dirichlet and one Neumann

# Create a mesh and define function sapaces
mesh = UnitSquareMesh(8, 8) 
    # Uniform finite element over a unit square
    # Divide the square in 8x8 rectangles, each divide in pair of triangles(Cells)

V = FunctionSpace(mesh, "P", 1) 
    # P - Especify the element type - in this case, Lagrange Family element
    # also can be specify as "Lagrange"
    # femtable.org (Periodic Table of the Finite Elements)
    # 1 - specify the degree of the finite element
    # 'DP' - creates a function space for discontinuous Galerkin method


# Define boundary condition
u_L = Expression("1 + 2*x[1]*x[1]", degree=2)

u_R = Expression("2 + 2*x[1]*x[1]", degree=2)
    # u_D -> Expression defining the solution values on the boundary
    # boudary -> is a function (or object) defining which points belong to the boundary
    # The formula must be written with C++ syntax 


tol = 1E+14
def boundary_L(x, on_boundary):
    return on_boundary and (near(x[0], 0, tol))

bc_L = DirichletBC(V, u_L, boundary_L)

def boundary_R(x, on_boundary):
    return on_boundary and (near(x[0], 1, tol))

bc_R = DirichletBC(V, u_R, boundary_R)

bcs = [bc_R, bc_L]

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(-6)

g = Expression('4*x[1]', degree=1)

a = dot(grad(u), grad(v))*dx  # For matrices is used "inner" insted of "dot". For vectors are equivalent
L = f*v*dx - g*v*ds
    # dx -> Differential element for integration over the domain
    # ds -> Denote integration over boundary


# Compute solution
u = Function(V) 
    # First u -> Defined as a TrialFunction and used it to represent the unknown in the form "a"
    # Redefined -> to be a Function object representing the solution
solve(a == L, u, bcs)

#Plot solution and mesh
plot(u)
plot(mesh)

#Save Solution to file in VTK format
# vtkfile = File('poisson/solution.pvd')
# vtkfile << u
    # Paraview are not working with .pvd file
xdmf_file = XDMFFile("poisson_MultDBC/solution_MultDBC.xdmf")
xdmf_file.write(u)
    # Using .xdmf insted

# Plot
# plt.show()
# plt.interactive(True) 
plt.savefig("poisson_MultDBC/poisson_MultDBC.png")