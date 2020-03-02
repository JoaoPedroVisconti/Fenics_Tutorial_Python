from fenics import *
import matplotlib as pltlib
pltlib.use('Agg')  # To be able to plot
import numpy as np
from ufl.operators import nabla_div
import matplotlib.pyplot as plt
from mshr import *

# Time
T = 1
num_steps = 500
dt = T / num_steps
eps = 0.01
K = 10

# Read mesh from file
mesh = Mesh('navier_stokes_cylinder/navier_stokes_cylinder.xml.gz')
# plot(mesh)
# plt.savefig('reaction_system/mesh.png') # Test to see the mesh

# Define the finite element function space for velocity
W = VectorFunctionSpace(mesh, 'P', 2)

# Define function space for system of concentration
P1 = FiniteElement('P', triangle, 1)
element = MixedElement([P1, P1, P1])
V = FunctionSpace(mesh, element)

# Define test functions
v_1, v_2, v_3 = TestFunctions(V)

# Define functions for velocity and concentration
w = Function(W)
u = Function(V)
u_n = Function(V)

# Split system functions to access components
u_1, u_2, u_3 = split(u)
u_n1, u_n2, u_n3 = split(u_n)

# Define source term
f_1 = Expression('pow(x[0]-0.1, 2) + pow(x[1]-0.1, 2) < 0.05*0.05 ? 0.1 : 0', degree=1)
f_2 = Expression('pow(x[0]-0.1, 2) + pow(x[1]-0.3, 2) < 0.05*0.05 ? 0.1 : 0', degree=1)
f_3 = Constant(0)

# Define expression used in variational form
k = Constant(dt)
K = Constant(K)
eps = Constant(eps)

# Define variational problem
F = ((u_1-u_n1) / k)*v_1*dx + dot(w, grad(u_1))*v_1*dx \
    + eps*dot(grad(u_1), grad(v_1))*dx + K*u_1*u_2*v_1*dx \
    + ((u_2-u_n2) / k)*v_2*dx + dot(w, grad(u_2))*v_2*dx \
    + eps*dot(grad(u_2), grad(v_2))*dx + K*u_1*u_2*v_2*dx \
    + ((u_3-u_n3) / k)*v_3*dx + dot(w, grad(u_3))*v_3*dx \
    + eps*dot(grad(u_3), grad(v_3))*dx - K*u_1*u_2*v_3*dx + K*u_3*v_3*dx \
    - f_1*v_1*dx - f_2*v_2*dx - f_3*v_3*dx

# Create time series for reading velocity data
timeseries_w = TimeSeries('navier_stokes_cylinder/velocity_series')

# Create XDMF files
xdmf_file_u_1 = XDMFFile("reaction_system/u_1.xdmf")
xdmf_file_u_2 = XDMFFile("reaction_system/u_2.xdmf")
xdmf_file_u_3 = XDMFFile("reaction_system/u_3.xdmf")

# Create progress bar
progress = Progress('Time-stepping', num_steps)
for i in range(num_steps):
    set_log_level(LogLevel.PROGRESS)
    progress+=1

# Time-stepping 
t = 0
for i in range(num_steps):

    # Update current time
    t += dt

    # Read velocity from file
    timeseries_w.retrieve(w.vector(), t)

    # Solve variational problem for time step
    solve(F == 0, u)

    # Save solution to file XDMF
    _u_1, _u_2, _u_3 = u.split()
    xdmf_file_u_1.write(_u_1, t)
    xdmf_file_u_2.write(_u_2, t)
    xdmf_file_u_3.write(_u_3, t)

    plt.figure(1)
    plt.subplot(1,3,1)
    plot(u_1, title='Concentration A')

    plt.subplot(1,3,2)
    plot(u_2, title='Concentration B')

    plt.subplot(1,3,3)
    plot(u_3, title='Concentration C')


    # Update previous solution
    u_n.assign(u)

# Plot
plt.savefig('reaction_system/solve.png')
