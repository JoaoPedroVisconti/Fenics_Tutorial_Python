from fenics import *
import matplotlib as pltlib
#pltlib.use('Agg')  # To be able to plot
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

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(-6)
a = dot(grad(u), grad(v))*dx  # For matrices is used "inner" insted of "dot". For vectors are equivalent
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
xdmf_file = XDMFFile("poisson/solution.xdmf")
xdmf_file.write(u)
    # Using .xdmf insted

# Compute error in L2 norm
error_L2 = errornorm(u_D, u, 'l2')

# Compute maximum error at vertices
vertex_values_u_D = u_D.compute_vertex_values(mesh)
vertex_values_u = u.compute_vertex_values(mesh) 
    # returns the value at all the vertices of the mesh as a numpy array

error_max = np.max(np.abs(vertex_values_u_D - vertex_values_u))

# Print errors
print(vertex_values_u[3]) # Access the value of the vertice
print("Error_L2 = ", error_L2)
print("Error_max = ", error_max)


# Plot
plt.show()
# plt.interactive(True) 
plt.savefig("poisson/poisson.png")