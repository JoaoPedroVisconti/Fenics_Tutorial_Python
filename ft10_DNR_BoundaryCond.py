from fenics import *
import matplotlib as pltlib
pltlib.use('Agg')  # To be able to plot
import numpy as np
from ufl.operators import nabla_div
import matplotlib.pyplot as plt
from mshr import *

import sympy as sym
# Define manufactured solution in sympy and derive f, g, etc.

x, y = sym.symbols('x[0], x[1]')            # needed by UFL
u = 1 + x**2 + 2*y**2                       # exact solution
u_e = u                                     # exact solution
u_00 = u.subs(x, 0)                         # restrict to x = 0
u_01 = u.subs(x, 1)                         # restrict to x = 1
f = -sym.diff(u, x, 2) - sym.diff(u, y, 2)  # -Laplace(u)
f = sym.simplify(f)                         # simplify f
g = -sym.diff(u, y).subs(y, 1)              # compute g = -du/dn
r = 1000                                    # Robin data, arbitrary
s = u                                       # Robin data, u = s

# Collect variables
variables = [u_e, u_00, u_01, f, g, r, s]

# Turn into C/C++ code strings
variables = [sym.printing.ccode(var) for var in variables]

# Turn into FEniCS Expressions
variables = [Expression(var, degree=2) for var in variables]

# Extract variables
u_e, u_00, u_01, f, g, r, s = variables

Nx = Ny = 8
mesh = UnitSquareMesh(Nx, Ny)

V = FunctionSpace(mesh, "P", 1)

u = TrialFunction(V)
v = TestFunction(V)

u_D = Expression('1 + 2*x[1]*x[1]', degree=2)

boundary_markers = MeshFunction("size_t", mesh, 1)


# Set up the boundaries and marking then
class Boundaryx0(SubDomain): # x = 0
    def inside(self, x, on_boundary):
        tol = 1e-14
        return on_boundary and near(x[0], 0, tol)

class Boundaryx1(SubDomain): # x = 1
    def inside(self, x, on_boundary):
        tol = 1e-14
        return on_boundary and near(x[0], 1, tol)

class Boundaryy0(SubDomain): # y = 0
    def inside(self, x, on_boundary):
        tol = 1e-14
        return on_boundary and near(x[1], 0, tol)

class Boundaryy1(SubDomain): # y = 1
    def inside(self, x, on_boundary):
        tol = 1e-14
        return on_boundary and near(x[1], 1, tol)

bx0 = Boundaryx0()
bx1 = Boundaryx1()
by0 = Boundaryy0()
by1 = Boundaryy1()

# Mark as subdomains 0, 1, 2, 3
bx0.mark(boundary_markers, 0)
bx1.mark(boundary_markers, 1)
by0.mark(boundary_markers, 2)
by1.mark(boundary_markers, 3)

boundary_conditions = {
    0 : {'Dirichlet': u_D},
    1 : {'Robin': (r,s)},
    2 : {'Neumann': g},
    3 : {'Neumann': 0} 
}

###################################################
# a Dirichlet condition u = u_D for x = 0
# a Robin condition k du/dn = r(u - s) for x = 1
# a Neumman cobdition -k du/dn = g for y = 0
# a Neumman cobdition -k du/dn = 0 for y = 1

# Collecting Dirichlet BC
bcs = []
for i in boundary_conditions:
    if 'Dirichlet' in boundary_conditions[i]:
        bc = DirichletBC(V, boundary_conditions[i]['Dirichlet'], boundary_markers, i)

        bcs.append(bc)


# Define the measure ds and dx in terms of our boundary markers
# This way ds(0) implies integration over subdomain 0 and so on
ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)
dx = Measure('dx', domain=mesh, subdomain_data=boundary_markers)


# Collecting all Neumann BC with the following snippet
integrals_N = []
for i in boundary_conditions:
    if 'Neumann' in boundary_conditions[i]:
        if boundary_conditions[i]['Neumann'] != 0:
            g = boundary_conditions[i]['Neumann']
            integrals_N.append(g*v*ds(i))

######### Can be use ############
# # Collecting all Robin BC
# integrals_R_a = []
# integrals_R_L = []
# for i in boundary_conditions:
#     if 'Robin' in boundary_conditions[i]:
#         r, s = boundary_conditions[i]['Robin']
#         integrals_R_a.append(r*u*v*ds(i))
#         integrals_R_L.append(r*s*v*ds(i))

# # Variational formulation
# a = kappa*dot(grad(u), grad(v))*dx + sum(integrals_R_a)
# L = f*v*dx - sum(integrals_N) + sum(integrals_R_L)

# Collecting all Robin BC
integrals_R = []
for i in boundary_conditions:
    if 'Robin' in boundary_conditions[i]:
        r, s = boundary_conditions[i]['Robin']
        integrals_R.append(r*(u - s)*v*ds(i))

kappa = Constant(1)

# Variational Formulation
F = kappa*dot(grad(u), grad(v))*dx + sum(integrals_R) - f*v*dx + sum(integrals_N)
a, L = lhs(F), rhs(F)

u = Function(V)

solve(a==L, u, bcs)

#Plot Solution
plot(u, title='Finite element solution')
plot(mesh)

plt.savefig("DNR_BoundaryCond/DNR_BoundaryCond.png")