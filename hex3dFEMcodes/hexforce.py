import numpy as np
from numpy.polynomial.legendre import leggauss
def gauss_quadrature(num_points):
    psi_values, weights_psi = leggauss(num_points)
    # eta_values, weights_eta = leggauss(num_points)
    return psi_values, weights_psi
def boundary_forces(limits,t,p):
    eta,w=gauss_quadrature(1)
    e=eta[0]
    w=w[0]
    z=0
    N=np.array([[(1-p)*(1-e)*(1-z)*(1/8),0,0],[0,(1-p)*(1-e)*(1-z)*(1/8),0],[0,0,(1-p)*(1-e)*(1-z)*(1/8)],
                [(1+p)*(1-e)*(1-z)*(1/8),0,0],[0,(1+p)*(1-e)*(1-z)*(1/8),0],[0,0,(1+p)*(1-e)*(1-z)*(1/8)],
                [(1+p)*(1-e)*(1-z)*(1/8),0,0],[0,(1+p)*(1-e)*(1-z)*(1/8),0],[0,0,(1+p)*(1-e)*(1-z)*(1/8)],
                [(1-p)*(1+e)*(1-z)*(1/8),0,0],[0,(1-p)*(1+e)*(1-z)*(1/8),0],[0,0,(1-p)*(1+e)*(1-z)*(1/8)],
                [(1-p)*(1-e)*(1+z)*(1/8),0,0],[0,(1-p)*(1-e)*(1+z)*(1/8),0],[0,0,(1-p)*(1-e)*(1+z)*(1/8)],
                [(1+p)*(1-e)*(1+z)*(1/8),0,0],[0,(1+p)*(1-e)*(1+z)*(1/8),0],[0,0,(1+p)*(1-e)*(1+z)*(1/8)],
                [(1+p)*(1+e)*(1+z)*(1/8),0,0],[0,(1+p)*(1+e)*(1+z)*(1/8),0],[0,0,(1+p)*(1+e)*(1+z)*(1/8)],
                [(1-p)*(1+e)*(1+z)*(1/8),0,0],[0,(1-p)*(1+e)*(1+z)*(1/8),0],[0,0,(1-p)*(1+e)*(1+z)*(1/8)]])
    [a,b]=limits
    J = np.abs(b - a) / 2
    integral=w*J*np.dot(N,t)
    return integral


def connect(force,global_force,elem):
    c = np.zeros(24)
    c[0:25:3] = 3*elem
    c[1:25:3] = 3*elem+1
    c[2:25:3] = 3*elem+2
    c= c.astype('int32')
    global_force[c,:] += force
    return global_force

def tractive_force_hex():
    nodes = np.array([
    [0, 0, 0],
    [2, 0, 0],
    [2, 2, 0],
    [0, 2, 0],
    [0, 0, 2],
    [2, 0, 2],
    [2, 2, 2],
    [0, 2, 2]])

# Example element connectivity for a cube
    elements = np.array([[0,1,2,3,4,5,6,7]])
    t=np.array([[2],[0],[0]])
    a=boundary_forces([0,2],t,p=1)
    # b=boundary_forces([1,2],t,psi=1)
    gf=np.zeros([3*len(nodes),1])
    # gf = connect(b,gf,elements[1])
    gf = connect(a,gf,elements[0])
    return gf
