from dolfin import *
import matplotlib as pltlib
pltlib.use('Agg')  # To be able to plot
import numpy as np
from ufl.operators import nabla_div
import matplotlib.pyplot as plt
from mshr import *

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
u_D = Expression("1 + x[0]*x[0] + 2*x[1]*x[1]", degree=2)
    # u_D -> Expression defining the solution values on the boundary
    # boudary -> is a function (or object) defining which points belong to the boundary
    # The formula must be written with C++ syntax 

def boundary(x, on_boundary):
    return on_boundary
    # boundary -> specifies which points that belong to the part of the boundary where the boundary
    # conditions should be applied
    # Must return a boolean value
    # true = given point x lies on the boundary
    # false = otherwise
    # Alternatively:
    # 
    # tol = 1E+14
    # def boundary(x):
    #   return near(x[0], 0, tol) or near(x[1], 0, tol) or near(x[0], 1, tol) or near(x[1], 1, tol) 

bc = DirichletBC(V, u_D, boundary)

tol = 1e-14
k_0 = 1
k_1 = 0.1
kappa = Expression('x[1] <= 0.5 + tol ? k_0 : k_1', degree=0, tol=tol, k_0=k_0, k_1=k_1)
    # Dividing the domain in two subdomains

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(-6)
a = kappa * dot(grad(u), grad(v))*dx  # For matrices is used "inner" insted of "dot". For vectors are equivalent
L = f*v*dx 
    # dx -> Differential element for integration over the domain
    # ds -> Later going to be use to denote integration over boundary


# Compute solution
u = Function(V) 
    # First u -> Defined as a TrialFunction and used it to represent the unknown in the form "a"
    # Redefined -> to be a Function object representing the solution
solve(a == L, u, bc)

#Plot solution and mesh
plot(u)
plot(mesh)

#Save Solution to file in VTK format
# vtkfile = File('poisson/solution.pvd')
# vtkfile << u
    # Paraview are not working with .pvd file
xdmf_file = XDMFFile("poisson_DiffMat/poisson_DiffMat_1.xdmf")
xdmf_file.write(u)
    # Using .xdmf insted

plt.savefig("poisson_DiffMat/poisson_DiffMat_1.png")