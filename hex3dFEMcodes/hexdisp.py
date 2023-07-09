import numpy as np
from hexahedral import connectivity
from hexforce import tractive_force_hex
def remove_row_and_column(displacements, nodes,elements,v, E,r):
    matrix = connectivity(nodes,elements,v,E,r)
    rows_to_remove = []
    for node, displacement in displacements.items():
    
        if displacement[0] is not None:
            i1 = node * 3 
            rows_to_remove.append(i1)
        if displacement[1] is not None:
            i2 = node * 3+1
            rows_to_remove.append(i2)
        if displacement[2] is not None:
            i3 = node * 3 +2 
            rows_to_remove.append(i3)
    
    new_matrix = np.delete(matrix, rows_to_remove, axis=0)  # remove rows
    new_matrix = np.delete(new_matrix, rows_to_remove, axis=1)  # remove columns
    
    return new_matrix
def remove_row(loads):

    f=tractive_force_hex()
    for node, load in loads.items():
        f[3 * node] = load[0]
        f[3 * node+1] = load[1]
        f[3 * node+2] = load[2] 
    rows_to_remove = []
    for node, force in loads.items():
        if force[0] is not None:
            i1 = node * 3
            rows_to_remove.append(i1)
        if force[1] is not None:
            i2 = node * 3+1
            rows_to_remove.append(i2)
        if force[2] is not None:
            i3 = node * 3+2
            rows_to_remove.append(i3)
    new_matrix = np.delete(f, rows_to_remove, axis=0)         
    return new_matrix
def disp(displacements,loads,nodes,elements,v,E,r):
    fr=remove_row(loads)
    kg=remove_row_and_column(displacements, nodes,elements,v, E,r)
    U=np.full([3*len(nodes),1],np.nan)
    for node, displacement in displacements.items():
        if displacement[0] is not None:
            i1=3*node
            U[i1]=displacement[0]
        if displacement[1] is not None:
            i2=3*node+1
            U[i2]=displacement[1]
        if displacement[2] is not None:
            i3=3*node+2
            U[i3]=displacement[2]
    free=np.argwhere(np.isnan(U))[:,0]
    U[[free],:] = np.linalg.solve(kg,fr) 
    return U
# nodes = np.array([
#     [0, 0, 0],
#     [2, 0, 0],
#     [2, 2, 0],
#     [0, 2, 0],
#     [0, 0, 2],
#     [2, 0, 2],
#     [2, 2, 2],
#     [0, 2, 2]])
# elements = np.array([[0,1,2,3,4,5,6,7]])
# v=0.3
# r=np.sqrt(1/3)
# E=1
# displacements={0:[0,0,0],3:[0,0,0],4:[0,0,0],7:[0,0,0]}
# loads={0:[0,0,0],3:[0,0,0],4:[0,0,0],7:[0,0,0]}
# K=connectivity(nodes, elements,v,E,[-r,r])
# u=(disp(displacements,loads,nodes,elements,v,E,[-r,r]))
# f=np.dot(K,u)
# print(f)
# print(u)