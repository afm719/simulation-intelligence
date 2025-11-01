""""
#############################
The pole balancing problem
#############################
Course: Simulation Intelligence and Autonomous Systems
Institution: Università degli studio di Trieste
Author: Arahí Fernández Monagas
Date: October 2025
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

sp.init_printing(use_unicode=True)


# Variables and states
x1 = sp.symbols('theta')      # pole angle with the vertical
x2 = sp.symbols('theta_dot')  # pole angular velocity
x3 = sp.symbols('pc')          # cart position
x4 = sp.symbols('pc_dot')      # cart velocity


x = sp.Matrix([x1, x2, x3, x4])  # state vector
u = sp.symbols('F')               # control input (force applied to the cart) 

g = sp.symbols('g')    # gravity acceleration
m_c = sp.symbols('m_c')  # cart mass
m_p = sp.symbols('m_p')  # pole mass
l = sp.symbols('l')    # half pole length
mu_p = sp.symbols('mu_p')  # pole friction coefficient

print("State vector (x):")
sp.pprint(x)
print("\Input (u):", u)


#Definition of the system
x1_dot = x2
x2_dot = (g*sp.sin(x1) + sp.cos(x1)*((-u - m_p*l*x2**2*sp.sin(x1) + mu_p*x2)/(m_c + m_p)) - (mu_p*x2)/(m_p*l)) / (l*(4/3 - (m_p*sp.cos(x1)**2)/(m_c + m_p)))
x3_dot = x4
x4_dot = (u + m_p*l*(x2**2*sp.sin(x1) - x2_dot*sp.cos(x1)) - mu_p*x4) / (m_c + m_p)

x_dot = sp.Matrix([x1_dot, x2_dot, x3_dot, x4_dot])
print("\nState derivatives (x_dot):")
sp.pprint(x_dot)


#Now I'll calculate the equilibrium point
x_eq = x_dot.subs([
    (u, 0),
    (x2, 0),
    (x4, 0)
])
sp.pprint(x_eq)

# Solve for equilibrium points
equilibrium_points = sp.solve([eq for eq in x_eq], (x1, x2, x3, x4))
print("\nEquilibrium points:")
sp.pprint(equilibrium_points)   

eq_1 = sp.Matrix([0,0,0,0])
eq_2 = sp.Matrix([sp.pi,0,0,0])
u_bar = 0

#Linearization around the equilibrium point
A = x_dot.jacobian(x)
B = x_dot.jacobian(sp.Matrix([u])).subs(equilibrium_points)
A_eq1 = A.subs([
    (x1, eq_1[0]),
    (x2, eq_1[1]),
    (x3, eq_1[2]),
    (x4, eq_1[3]),
    (u, u_bar)
])
A_eq2 = A.subs([
    (x1, eq_2[0]),
    (x2, eq_2[1]),
    (x3, eq_2[2]),
    (x4, eq_2[3]),
    (u, u_bar)
])  
B_eq1 = B.subs([
    (x1, eq_1[0]),
    (x2, eq_1[1]),
    (x3, eq_1[2]),
    (x4, eq_1[3]),
    (u, u_bar)
])
B_eq2 = B.subs([
    (x1, eq_2[0]),
    (x2, eq_2[1]),
    (x3, eq_2[2]),
    (x4, eq_2[3]),
    (u, u_bar)
])  
print("\nLinearized A matrix around equilibrium point 1 (downward position):")
sp.pprint(A_eq1)
print("\nLinearized B matrix around equilibrium point 1 (downward position):")
sp.pprint(B_eq1)
print("\nLinearized A matrix around equilibrium point 2 (upright position):")
sp.pprint(A_eq2)
print("\nLinearized B matrix around equilibrium point 2 (upright position):")
sp.pprint(B_eq2)


#Simulation
