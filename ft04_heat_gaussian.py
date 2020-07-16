from fenics import *
import matplotlib.pyplot as plt
import numpy as np

T = 2                   # Final time
num_steps = 10          # Number of steps
dt = T/num_steps        # Time step size

nx = ny = 30
mesh = RectangleMesh(Point(-2,-2), Point(2,2), nx, ny) # No longer a unitsquare mesh
V = FunctionSpace(mesh, 'P', 1)

def boundary (x, on_boundary):
    return on_boundary

u_D = Constant(0) # Values on boundary

bc = DirichletBC(V, u_D, boundary)

# Define initial values
u_0 = Expression('exp(-a*pow(x[0], 2) - a*pow(x[1], 2))', degree=2, a=5) # Initial value

u_n = interpolate(u_0, V)

# Define Variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(0)  # ?????

F = u*v*dx + dt*dot(grad(u), grad(v))*dx - (u_n + dt*f)*v*dx
a, L = lhs(F), rhs(F)

xdmf_file_u = XDMFFile("heat_gaussian/heat_gaussian.xdmf")

# Time-stepping 
u = Function(V)
t = 0
for n in range(num_steps):

    t+=dt

    solve(a==L, u, bc)

    xdmf_file_u.write(u, t)

    plot(u)

    u_n.assign(u) # Updating the previous step

plt.show()
plt.savefig("heat_gaussian/heat_gaussian.png")

