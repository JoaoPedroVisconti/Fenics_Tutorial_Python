from fenics import *
import matplotlib as pltlib
pltlib.use('Agg')  # To be able to plot
import numpy as np
from ufl.operators import nabla_div
import matplotlib.pyplot as plt
from mshr import *
import sys

mesh = UnitSquareMesh(8, 8)

V = FunctionSpace(mesh, "P", 1) 

# Define boundary condition
u_D = Expression("1 + x[0]*x[0] + 2*x[1]*x[1]", degree=2)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_D, boundary)


    # def boundary(x, on_boundary):
    #     tol = 1e-14
    #     return on_boundary and near(x[0], 0, tol)

    ######## This is the same thing as #######
    # class Boundary(SubDomain):
    #     def inside(self, x, on_boundary):
    #         tol = 1e-14
    #         return on_boundary and near(x[0], 0, tol)

    # boundary = Boundary()
    # bc = DirichletBC(V, Constant(0), boundary)

tol = 1e-14
k_0 = 1
k_1 = 0.1
class Omega_0(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] <= 0.5 + tol

class Omega_1(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] >= 0.5 - tol

# Create a MeshFunction with non-negative (size_t) values
materials = MeshFunction('size_t', mesh, 2)
# Mark the cells belonging to each subdomain
subdomain_0 = Omega_0()
subdomain_1 = Omega_1()
subdomain_0.mark(materials, 0) # Set the value of the mesh function 'materials' to 0
subdomain_1.mark(materials, 1) # Set the value of the mesh function 'materials' to 1

# Test the mesh
# plot (materials)
# plt.savefig('poisson_DiffMat/mesh_test_2.jpg')

# Store the mesh function
# File('poisson_DiffMat/material_2.xml.gz') << materials

class K(Expression):
    def __init__(self, materials, k_0, k_1, **kwargs):
        self.materials = materials
        self.k_0 = k_0
        self.k_1 = k_1

def eval_cell(self, values, x, cell):
    if self.materials[cell.index] == 0:
        values[0] = self.k_0
    else:
        values[0] = self.k_1

kappa = K(materials, k_0, k_1, degree=0)

kappa = K(degree=0)
kappa.set_k_values(1, 0.01)
# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(-6)

a = kappa * dot(grad(u), grad(v))*dx
L = f*v*dx 

# Compute solution
u = Function(V) 

solve(a == L, u, bc)

#Plot solution and mesh
plot(u)
plot(mesh)



xdmf_file = XDMFFile("poisson_DiffMat/poisson_DiffMat_2.xdmf")
xdmf_file.write(u)

# Plot
plt.savefig("poisson_DiffMat/poisson_DiffMat_2.png")