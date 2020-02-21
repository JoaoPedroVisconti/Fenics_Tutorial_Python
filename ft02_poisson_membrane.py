from fenics import *
import matplotlib.pyplot as plt
import numpy as np
from mshr import *

# Create a mesh and define function sapaces
domain = Circle(Point(0, 0), 1)
mesh = generate_mesh(domain, 64)
    # The Circle shape from mshr takes the center and radius
    # generate_mesh function specifies the desire mesh resolution

beta = 8
R_0 = 0.6
p = Expression('4*exp(-pow(beta, 2)*(pow(x[0], 2) + pow(x[1] - R_0, 2)))', degree=1, beta=beta, R_0=R_0)

V = FunctionSpace(mesh, "P", 2) 
    # P - Especify the element type - in this case, Lagrange Family element
    # also can be specify as "Lagrange"
    # femtable.org (Periodic Table of the Finite Elements)
    # 2 - specify the degree of the finite element
    # 'DP' - creates a function space for discontinuous Galerkin method


# Define boundary condition
w_D = Constant(0)
    # u_D -> Expression defining the solution values on the boundary
    # boudary -> is a function (or object) defining which points belong to the boundary

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

bc = DirichletBC(V, w_D, boundary)

# Defining the load


# Define variational problem
w = TrialFunction(V)
v = TestFunction(V)
a = dot(grad(w), grad(v))*dx  # For matrices is used "inner" insted of "dot". For vectors are equivalent
L = p*v*dx
    # dx -> Differential element for integration over the domain
    # ds -> Later going to be use to denote integration over boundary


# Compute solution
w = Function(V) 
    # First u -> Defined as a TrialFunction and used it to represent the unknown in the form "a"
    # Redefined -> to be a Function object representing the solution
solve(a == L, w, bc)

#Plot solution and mesh
p = interpolate(p, V)
    # Trasnform the formula (Expression) to a finite element function (Function).
plt.figure(1)
plt.subplot(1,2,1)
plot(w, title='Deflection')
plt.subplot(1,2,2)
plot(p, title='Load')
plt.savefig("poisson_membrane/membrane.png")

xdmf_file_u = XDMFFile("poisson_membrane/deflection.xdmf")
xdmf_file_u.write(w)
xdmf_file_p = XDMFFile("poisson_membrane/load.xdmf")
xdmf_file_p.write(p)

tol = 0.001 # To avoid hitting points outside the domain

y = np.linspace(-1 + tol, 1 - tol, 101)

points = [(0, y_) for y_ in y] #2D points

w_line = np.array([w(point) for point in points])
p_line = np.array([p(point) for point in points])

plt.figure(2)
plt.plot(y, 50*w_line, 'k', linewidth=2)
plt.plot(y, p_line, 'b--', linewidth=2)
plt.grid(True)
plt.xlabel('$y$')
plt.legend(['Deflection ($\\times 50$)', 'Load'], loc='upper left')
plt.savefig('poisson_membrane/curves.png')

