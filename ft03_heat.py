# Test Problem 1: A Known analytical solution
from fenics import *
# import matplotlib as pltlib
# pltlib.use('Agg') # To be able to plot
import matplotlib.pyplot as plt
from ufl.operators import nabla_div
import numpy as np


T = 2
num_steps = 10
dt = T/num_steps

nx = ny = 8
mesh = UnitSquareMesh(nx, ny)
V = FunctionSpace(mesh, 'P', 1)

# Expression for boundary condition
alpha = 3; beta = 1.2
u_D = Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t', degree=2, alpha=alpha, beta=beta, t=0)
    # Expression with time t as a parameter
    # The time t is going to be updated later

# Defining boundary condition
def boundary (x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_D, boundary)

# Shall use  u = u.n+1 and u_n = u.n
# Since we set the t = 0 for the boundary value u_D, this can be used for the initial condition
# u_n = project(u_D, V) 
    # Initial value can be calculated by projection or
u_n = interpolate(u_D, V) 
    # by interpolation

# Variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(beta - 2 - 2*alpha)

# Set function. Fenics determine a and L
F = u*v*dx + dt*dot(grad(u), grad(v))*dx - (u_n + dt*f)*v*dx
a, L = lhs(F), rhs(F)

# Performing the time-stepping loop
u = Function(V)
t = 0
for n in range (num_steps):
    
    #Update the current time
    t += dt
    u_D.t = t

    #Solve variational problem
    solve(a == L, u, bc)

    # Update previous solution
    u_n.assign(u)
        # Must be done using the "assign". Need two variables (previous and current time step)
    
    plot(u)

    # Compute the error at vertices
    u_e = interpolate(u_D, V)
    error = np.abs(u_e.vector().get_local() - u.vector().get_local()).max()
    print('t = %.2f: error = %.3g' % (t, error))

plt.show()

plt.savefig('heat/heat.png')
