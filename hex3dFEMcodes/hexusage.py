import numpy as np
from hex3dglobalmatrix import connectivity
from hexdisp import disp

nodes = np.array([
    [0, 0, 0],
    [2, 0, 0],
    [2, 2, 0],
    [0, 2, 0],
    [0, 0, 2],
    [2, 0, 2],
    [2, 2, 2],
    [0, 2, 2]])
elements = np.array([[0,1,2,3,4,5,6,7]])
v=0.3
r=np.sqrt(1/3)
E=1
displacements={0:[0,0,0],3:[0,0,0],4:[0,0,0],7:[0,0,0]}
loads={0:[0,0,0],3:[0,0,0],4:[0,0,0],7:[0,0,0]}
K=connectivity(nodes, elements,v,E,[-r,r])
u=(disp(displacements,loads,nodes,elements,v,E,[-r,r]))
f=np.dot(K,u)
print(f)
print(u)