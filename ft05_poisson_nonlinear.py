from fenics import *
import matplotlib.pyplot as plt
import  numpy as np
import sympy as sym  # Have to import after fenics import

# Example taking the q(u)=1 * u^2
# SymPy -> for symbolic computing and integrate such computation
# diff - Symbolic defferentiation and 
# ccode - for C/C++ code generation

def q(u):  # Override q(u) from fenics import
    "Return nonlinear coefficient"
    return 1 + u**2

mesh = UnitSquareMesh(8, 8)
V = FunctionSpace(mesh, 'P', 1)


# Use SymPy to compute f from the manufactured solution u
x, y = sym.symbols('x[0], x[1]')
    # Define the names x and y as x[0] and x[1]. Valid syntax fro FEniCS

u = 1 + x + 2*y
f = - sym.diff(q(u)*sym.diff(u, x), x) - sym.diff(q(u)*sym.diff(u, y), y)
f = sym.simplify(f)

# Two parts to turning the expression for u and f into C or C++  syntax for FEniCS "Expression"
# 1 -> Ask for the C code of the expressions
u_code = sym.printing.ccode(u)
f_code = sym.printing.ccode(f)

print("u = ", u_code)
print("f = ", f_code)

u_D = Expression(u_code, degree=1)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_D, boundary)

u = Function(V)
v = TestFunction(V)
f = Expression(f_code, degree=1)
F = q(u)*dot(grad(u), grad(v))*dx - f*v*dx

solve(F == 0, u, bc)

plot(u)

plt.savefig('nonlinear_poisson/nonlinear_poisson.png')
