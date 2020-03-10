from fenics import *
import matplotlib as pltlib
pltlib.use('Agg')  # To be able to plot
import numpy as np
from ufl.operators import nabla_div
import matplotlib.pyplot as plt
from mshr import *

def solver(f, u_D, Nx, Ny, degree=1):
    """
    Solve -Laplace(u) = f on [0,1] x [0,1] with 2*Nx*Ny Lagrange
    elements of specified degree and u = u_D (Expression) on the 
    boundary
    """
    # Create mesh and define function space
    mesh = UnitSquareMesh(Nx, Ny)
    V = FunctionSpace(mesh, "P", degree)

    # Define boundary
    def boundary(x, on_boundary):
        return on_boundary
    
    # Boundari conditions
    bc = DirichletBC(V, u_D, boundary)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    a = dot(grad(u), grad(v))*dx
    L = f*v*dx

    # Compute solution
    u = Function(V)
    solve(a==L, u, bc)

    return u

def run_solver():
    "Run solver to compute and post-process solution"

    # Set up problem parameters and call solver
    u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)
    f = Constant(-6)
    u = solver(f, u_D, 8, 8, 1)

    # Plot solution and mesh
    plot(u)
    plot(u.function_space().mesh())

    # Save in XDMF
    xdmf_file = XDMFFile("poisson_solver/solution.xdmf")
    xdmf_file.write(u)

if __name__ == "__main__":
    run_solver()