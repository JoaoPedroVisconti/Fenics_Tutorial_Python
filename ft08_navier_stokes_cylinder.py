from fenics import *
import matplotlib as pltlib
pltlib.use('Agg')  # To be able to plot
import numpy as np
from ufl.operators import nabla_div
import matplotlib.pyplot as plt
from mshr import *

T = 0.1
num_steps = 500
dt = T / num_steps
mu = 0.001
rho = 1

# Create the mesh
channel = Rectangle(Point(0, 0), Point(2.20, 0.41))
cylinder = Circle(Point(0.2, 0.2), 0.05)
domain = channel - cylinder  # Subtracting the two shapes

mesh = generate_mesh(domain, 64) # mesh wiht 64 cells across its diameter(channel length)
# plot(mesh)
# plt.savefig('testeMesh.png')

# Define function spaces
V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)

# Define Boundary
walls = 'near(x[1], 0) || near(x[1], 0.41)'
inflow = 'near(x[0], 0)'
outflow = 'near(x[0], 2.2)'
cylinder = 'on_boundary && x[0]>0.1 && x[0]<0.3 && x[1]>0.1 && x[1]<0.3'

# Define Inflow profile
inflow_profile = ('4 * 1.5 * x[1] * (0.41 - x[1]) / pow(0.41, 2)', '0')

# Define Boundary conditions
bcu_inflow = DirichletBC(V, Expression(inflow_profile, degree=2), inflow)
bcu_walls = DirichletBC(V, Constant((0, 0)), walls)
bcu_cylinder = DirichletBC(V, Constant((0, 0)), cylinder)
bcp_outflow = DirichletBC(Q, Constant(0), outflow)

bcu = [bcu_walls, bcu_inflow, bcu_cylinder]
bcp = [bcp_outflow]

# Define trial and test functions
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

# Define functions for solutions at previous and current time step
u_n = Function(V)
u_ = Function(V)
p_n = Function(Q)
p_ = Function(Q)

# Define expressions used in variational forms
U = 0.5*(u_n + u)
n = FacetNormal(mesh)
f = Constant((0, 0))
k = dt
mu = Constant(mu)
rho = Constant(rho)

# Define symmetric Gradient
def epsilon(u):
    return sym(nabla_grad(u))

# Define stress tensor
def sigma(u, p):
    return 2 * mu * epsilon(u) - p * Identity(len(u))

# Step 1
F1 = rho * dot((u - u_n)/k, v) * dx + \
    rho * dot(dot(u_n, nabla_grad(u_n)), v) * dx + \
    inner(sigma(U, p_n), epsilon(v)) * dx + \
    dot(p_n*n, v) * ds - \
    dot(mu * nabla_grad(U)*n, v) * ds - \
    dot(f, v) * dx

a1 = lhs(F1)
L1 = rhs(F1)

# Step 2
F2 = dot(nabla_grad(p), nabla_grad(q)) * dx - \
    dot(nabla_grad(p_n), nabla_grad(q)) * dx + \
    (1/k) * div(u_) * q * dx

a2 = lhs(F2)
L2 = rhs(F2)

# Step 3
F3 = dot(u, v) * dx - dot(u_, v) * dx + k * dot(nabla_grad(p_ - p_n), v) * dx

a3 = lhs(F3)
L3 = rhs(F3)

# For the split scheme we will first assemble and than call solve
# Assemble matrices
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)

# Apply boundary conditions to matrices
[bc.apply(A1) for bc in bcu]
[bc.apply(A2) for bc in bcp]

# Create XDMF filesfor visualization output
xdmf_file_u = XDMFFile("navier_stokes_cylinder/navier_stokes_cylinder_u.xdmf")
xdmf_file_p = XDMFFile("navier_stokes_cylinder/navier_stokes_cylinder_p.xdmf")

# Create time series (for use in reaction_system.py)
# timeseries_u = TimeSeries('navier_stokes_cylinder/navier_stokes_cylinder_u')
# timeseries_p = TimeSeries('navier_stokes_cylinder/navier_stokes_cylinder_p')

# Save mesh to file (for use in reaction_system.py)
File('navier_stokes_cylinder/navier_stokes_cylinder.xml.gz') << mesh

# Create Progress bar
progress = Progress('Time-stepping', num_steps)
for i in range(num_steps):
    set_log_level(LogLevel.PROGRESS)
    progress+=1

# Time-stepping
t = 0
for n in range(num_steps):

    # Update current time
    t += dt

    # Step 1: Tentative velocity step
    b1 = assemble(L1)
    [bc.apply(b1) for bc in bcu]
    solve(A1, u_.vector(), b1, 'bicgstab', 'hypre_amg')

    # Step 2: Pressure correction step
    b2 = assemble(L2)
    [bc.apply(b2) for bc in bcp]
    solve(A2, p_.vector(), b2, 'bicgstab', 'hypre_amg')

    # Step 3: Velocity correction step
    b3 = assemble(L3)
    solve(A3, u_.vector(), b3, 'cg', 'sor') 

    # Plot solution
    plot(u_, title='Velocity')
    plot(p_, title='Pressure')

    # Save solution to file (XDMF/HDF5)
    xdmf_file_u.write(u_, t)
    xdmf_file_p.write(p_, t)

    # timeseries_u.store(u_.vector(), t)
    # timeseries_p.store(p_.vector(), t)

    # Update previous solution
    u_n.assign(u_)
    p_n.assign(p_)

    # Update Progress bar
    # progress.Update(t / T)
    print('u max:', u_.vector().get_local().max())

# Save figure
plt.savefig('navier_stokes_cylinder/navier_stokes_cylinder.png')