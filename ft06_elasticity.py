from fenics import *
import matplotlib as pltlib
pltlib.use('Agg') # To be able to plot
import matplotlib.pyplot as plt
from ufl.operators import nabla_div
import numpy as np

#Scaled variables 
L = 1
W = 0.2
mu = 1
rho = 1
delta = W/L
gamma = 0.4*delta**2
beta = 1.25
lambda_ = beta
g = gamma

# Create a mesh and define Function Space
mesh = BoxMesh(Point(0, 0, 0), Point(L, W, W), 10, 3, 3)
V = VectorFunctionSpace(mesh, 'P', 1)
    # The primary unknown is now a vector field "u" and not a scalar field

# Define Boundary conditions
tol =1E-14

def clamped_boundary(x, on_boundary):
    return on_boundary and x[0] < tol


bc = DirichletBC(V, Constant((0, 0, 0)), clamped_boundary)

# Define Strain and Stress 
def epsilon(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)
    # return sym(nabla_grad(u))

def sigma(u):
    return lambda_*nabla_div(u)*Identity(d) + 2*mu*epsilon(u)

# Define Variational problem
u = TrialFunction(V)
d = u.geometric_dimension()
v = TestFunction(V)
f = Constant((0, 0, -rho*g))
T = Constant((0, 0, 0))
a = inner(sigma(u), epsilon(v))*dx
L = dot(f, v)*dx + dot(T, v)*dx

# Compute solution
u = Function(V)
    # We get "u" as a vector-value finite element function with three components
solve(a==L, u, bc)
plot(u, title="Displacement", mode="displacement")

xdmf_file_u = XDMFFile("elasticity/displacement.xdmf") # It's better to export the result as xdml file
xdmf_file_u.write(u)

# Plot Stress
s = sigma(u) - (1/3)*tr(sigma(u))*Identity(d)  # Deviatoric stress
von_Mises = sqrt(3/2*inner(s, s))
V = FunctionSpace(mesh, "P", 1)
    # von_Mises variable is now an expression that we must projected to a finite
    # element space before we can visualize it
von_Mises = project(von_Mises, V)
plt.figure(1)
plot(von_Mises, title='Stress intensity')
plt.savefig("elasticity/stress_vonMises.png")
xdmf_file_von = XDMFFile("elasticity/von_Mises.xdmf")
xdmf_file_von.write(von_Mises)

# Compute magnitude of  displacement
u_magnitude = sqrt(dot(u, u))
u_magnitude = project(u_magnitude, V)
plt.figure(2)
plot(u_magnitude, "Displacement magnitude")
plt.savefig("elasticity/displacement_magnitude.png")
xdmf_file_mag = XDMFFile("elasticity/displacement_magnitude.xdmf")
xdmf_file_mag.write(u_magnitude)

print('min/max u:', u_magnitude.vector().get_local().min(), u_magnitude.vector().get_local().max())