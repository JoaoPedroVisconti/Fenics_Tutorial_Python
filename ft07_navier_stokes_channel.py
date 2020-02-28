from fenics import *
import matplotlib as pltlib
pltlib.use('Agg') # To be able to plot
import matplotlib.pyplot as plt
from ufl.operators import nabla_div
import numpy as np

T = 10                # Final time
num_steps = 500       # Number of steps
dt = T/num_steps      # Time step size
mu = 1                # Kinematic visconsity
rho = 1               # Density

# Create a mesh
mesh = UnitSquareMesh(8, 8)

V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)
# Need to define two FunctionSpace oone for velocity and other for pressure

# Define three boundary conditions
# u = 0 at the walls -> y=0 and y=1
walls = 'near(x[1], 0) || near(x[1], 1)'
# p = 8 at the inflow -> x=0
inflow = 'near(x[0], 0)'
# p = 0 at the outflow -> x=1
outflow = 'near(x[0], 1)'

# Define values of boundaries
bcu_noslip = DirichletBC(V, Constant((0, 0)), walls)
bcp_inflow = DirichletBC(Q, Constant(8), inflow)
bcp_outflow = DirichletBC(Q, Constant(0), outflow)

# Collect the boundary conditions for the velocity and pressure in Python lists
bcu = [bcu_noslip]
bcp = [bcp_inflow, bcp_outflow]

# Definition of the variational forms
# Three variational problems to be defined, one for each step

# Two sets of trial and test functions
u = TrialFunction(V)  # Unknown u^n+1
v = TestFunction(V)
p = TrialFunction(Q)  # Unknown p^n+1
q = TestFunction(Q)

# Define functions for solutions at previous and current time steps
u_n = Function(V)  # u^n
# The most recently computed approximation | u^n+1 available as Function object
u_ = Function(V)
p_n = Function(Q)  # p^n
# The most recently computed approximation | p^n+1 available as Function object
p_ = Function(Q)

# Constants
U = 0.5*(u_n + u)
n = FacetNormal(mesh)
f = Constant((0, 0))
k = Constant(dt)
mu = Constant(mu)
rho = Constant(rho)

# Define strain-rate tensor


def epsilon(u):
    return sym(nabla_grad(u))  # 0.5*(nabla_grad(u) + nabla_grad(u).T)

# Define stress tensor


def sigma(u, p):
    return 2*mu*epsilon(u) - p*Identity(len(u))


# Step 1
F1 = rho * dot((u - u_n)/k , v) * dx + \
    rho * dot(dot(u_n, nabla_grad(u_n)), v) * dx + \
    inner(sigma(U, p_n), epsilon(v))* dx + \
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

# Time-stepping
t = 0
for n in range(num_steps):

    # Update current time
    t += dt

    # Step 1: Tentative velocity step
    b1 = assemble(L1)
    [bc.apply(b1) for bc in bcu]
    solve(A1, u_.vector(), b1)

    # Step 2: Pressure correction step
    b2 = assemble(L2)
    [bc.apply(b2) for bc in bcp]
    solve(A2, p_.vector(), b2)

    # Step 3: Velocity correction step
    b3 = assemble(L3)
    solve(A3, u_.vector(), b3)

    # Plot solution
    plot(u_)

    # Compute error
    u_e = Expression(('4*x[1]*(1.0 - x[1])', '0'), degree=2)
    u_e = interpolate(u_e, V)
    error = np.abs(u_e.vector().get_local() - u_.vector().get_local()).max()
    #print('t = %.2f: error = %.3g' % (t, error))
    #print('max u:', u_.vector().get_local().max())

    # Update previous solution
    u_n.assign(u_)
    p_n.assign(p_)

# Save figure
plt.savefig('navier_stokes_channel/navier_stokes_channel.png')
