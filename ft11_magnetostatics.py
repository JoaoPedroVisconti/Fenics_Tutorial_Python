from fenics import *
import matplotlib as pltlib
pltlib.use('Agg')  # To be able to plot
import numpy as np
from ufl.operators import nabla_div
import matplotlib.pyplot as plt
from mshr import *
from math import cos, sin, pi

######### OBS ###########
# a and b -> the inner and outer radius of the iron cylinder
# c1 and c2 -> the radius of two concentric distribution of copper wire cross-sections
# r -> the radius of a copper wire
# R -> radius of the domain
# n -> number of windings (Given a total of 2n copper-wire cross-sections)
# Using mshr

# Constants
a = 1
b = 2
c_1 = 0.8
c_2 = 1.4
r = 0.1
R = 5
n = 10

# Define geometry for background
domain = Circle(Point(0, 0), R)

# Define geometry for iron cylinder
cylinder = Circle(Point(0, 0), b) - Circle(Point(0, 0), a)

# Define geometry for wires (N = North (up), S = South (down))
angles_N = [i*2*pi/n for i in range(n)]
angles_S = [(i + 0.5)*2*pi/n for i in range(n)]

wires_N = [Circle(Point(c_1*cos(v), c_1*sin(v)), r) for v in angles_N]
wires_S = [Circle(Point(c_2*cos(v), c_2*sin(v)), r) for v in angles_S]

# Mesh will be define for the entire disk i=with radius R but 
# need the mesh generation to respect the internal boundaries 
# defined by the iron cylinder and the copper wires.
# Also want the mesh to label the sub domain

# Set subdomains for iron cylinder
domain.set_subdomain(1, cylinder)

# Set subdomains for wires
for (i, wire) in enumerate(wires_N):
    domain.set_subdomain(2 + i, wire)

for (i, wire) in enumerate(wires_S):
    domain.set_subdomain(2 + n + i, wire)

# Once the subdomain have been created, we can generate the mesh
# Generete mesh
mesh = generate_mesh(domain, 128)
plot(mesh)
plt.savefig('meshtest.png')

V = FunctionSpace(mesh, "P", 1)

# Define boundary condition
bc = DirichletBC(V, Constant(0), 'on_boundary')

# Create a MeshFunction that marks the subdomains
markers = MeshFunction('size_t', mesh, 2, mesh.domains())
    # This create a MeshFunction with unsigned integer values
    # (the subdomain numbers) with dimension 2, which is the
    # cell dimension for this 2D problem.

# Now can use markers to redefine the integration measure dx
dx = Measure('dx', domain=mesh, subdomain_data=markers)
    # Integrals over subdomain can be express by x(0), x(1) ...

# Define current densities
J_N = Constant(1)
J_S = Constant(-1)

# Define magnetic permeability
class Permeability(UserExpression): # Use UserExpression inted of Expression
    
    def __init__(self, markers, **kwargs):
        super().__init__(**kwargs)

    # def __init__(self, markers, **kwargs):
        self.markers = markers 
    
    def eval_cell(self, values, x, cell):
        if self.markers[cell.index] == 0:
            values[0] = 4*pi*1e-7   # Vacuum
        elif self.markers[cell.index] == 1:
            values[0] = 1e-5  # Iron (should really be 6.3e-3)
        else:
            values[0] = 1.26e-6  # Copper

mu = Permeability(markers, degree=1)

# Define Variational problem
A_z = TrialFunction(V)
v = TestFunction(V)

a = (1 / mu)*dot(grad(A_z), grad(v))*dx
L_N = sum(J_N*v*dx(i) for i in range(2, 2 + n))
L_S = sum(J_S*v*dx(i) for i in range(2 + n, 2 + 2*n))
L = L_N + L_S

# Solve variational problem
A_z = Function(V)
solve(a == L, A_z, bc)

# Compute magnetic field
W = VectorFunctionSpace(mesh, "P", 1)
B = project(as_vector((A_z.dx(1), -A_z.dx(0))), W)

# Plot solution
plt.figure(1)
plt.subplot(2,1,1)
plot(A_z)
plt.subplot(2,1,2)
plot(B)


plt.savefig('magnetostatics/magnetostatics.png')
# Save solution on XDML file
