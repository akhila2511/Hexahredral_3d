import numpy as np
import pandas as pd
from hexforce import tractive_force_hex
from numpy.polynomial.legendre import leggauss

def stiffness_matrix(v,E,x1, x2, x3, x4, x5, x6, x7, x8,y1, y2, y3, y4,y5, y6, y7, y8,z1, z2, z3, z4, z5, z6, z7, z8,r):

    def gauss_quadrature(num_points):
        psi_values, weights_psi = leggauss(num_points)
        eta_values, weights_eta = leggauss(num_points)
        zeta_values, weights_zeta = leggauss(num_points)
        return psi_values, eta_values, zeta_values, weights_psi, weights_eta, weights_zeta

    psi_values, eta_values, zeta_values, weights_psi, weights_eta, weights_zeta = gauss_quadrature(2)
    K_local = np.zeros([24,24])
    D=np.array([[1-v,v,v,0,0,0],[v,1-v,v,0,0,0],[v,v,1-v,0,0,0],
                [0,0,0,(1-2*v)/2,0,0],[0,0,0,0,(1-2*v)/2,0],[0,0,0,0,0,(1-2*v)/2]])*(E/((1+v)*(1-2*v)))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                # p, e, z = psi_values[i], eta_values[j], zeta_values[k]
                p,e,z=r[i],r[j],r[k]
                w1,w2,w3=weights_psi[i],weights_eta[j],weights_zeta[k]
                dN = np.array([[-(1 - e) * (1 - z), (1 - e) * (1 - z), (1 + e) * (1 - z), -(1 + e) * (1 - z),
                                -(1 - e) * (1 + z), (1 - e) * (1 + z), (1 + e) * (1 + z), -(1 + e) * (1 + z)],
                               [-(1 - p) * (1 - z), -(1 + p) * (1 - z), (1 + p) * (1 - z), (1 - p) * (1 - z),
                                -(1 - p) * (1 + z), -(1 + p) * (1 + z), (1 + p) * (1 + z), (1 - p) * (1 + z)],
                               [-(1 - p) * (1 - e), -(1 + p) * (1 - e), -(1 + p) * (1 + e), -(1 - p) * (1 + e),
                                (1 - p) * (1 - e), (1 + p) * (1 - e), (1 + p) * (1 + e), (1 - p) * (1 + e)]])*(1/8)
                a = np.array([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3], [x4, y4, z4],
                              [x5, y5, z5], [x6, y6, z6], [x7, y7, z7], [x8, y8, z8]])
                J = np.dot(dN, a)
                B = np.dot(np.linalg.inv(J), dN)
                b1 = np.zeros([6, 24])

                for m in range(8):
                    b1[0, 3 * m] = B[0][m]
                    b1[1, 3 * m + 1] = B[1][m]
                    b1[2, 3 * m + 2] = B[2][m]
                    b1[3, [3 * m + 1, 3 * m + 2]] = [B[2][m], B[1][m]]
                    b1[4, [3 * m, 3 * m + 2]] = [B[2][m], B[0][m]]
                    b1[5, [3 * m, 3 * m + 1]] = [B[1][m], B[0][m]]

                J1=np.linalg.det(J)
                K_local+=w1*w2*w3*J1*np.dot(b1.T,(np.dot(D,b1)))
    return K_local


def connectivity(nodes, elements,v,E,r):
    K_global = np.zeros([3 * len(nodes), 3 * len(nodes)])
    for i in range(len(elements)):
        x1, x2, x3, x4, x5, x6, x7, x8 = nodes[elements[0]][:, 0]
        y1, y2, y3, y4, y5, y6, y7, y8 = nodes[elements[0]][:, 1]
        z1, z2, z3, z4, z5, z6, z7, z8 = nodes[elements[0]][:, 2]

        K_local = stiffness_matrix(v,E,x1, x2, x3, x4, x5, x6, x7, x8,y1, y2, y3, y4,y5, y6, y7, y8,z1, z2, z3, z4, z5, z6, z7, z8,r)
        c = np.zeros(24)
        c[0:22:3] = elements[i] * 3
        c[1:23:3] = elements[i] * 3 + 1
        c[2:24:3] = elements[i] * 3 + 2
        c = c.astype("int32").flatten()
        K_global[c[:, None], c] += K_local

    return K_global

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
df=pd.DataFrame()
df1=pd.DataFrame()
df2=pd.DataFrame()
df3=pd.DataFrame()
v=0.3
E=1e7
r=[0,1e-5,1e-4,1e-3,1e-2,1e-1,0.2,0.3,0.4,0.57735,0.7,1]
for i in range(len(r)):
    matrix = connectivity(nodes, elements,v,E,[-r[i],r[i]])
    A=np.linalg.eigvals(matrix)
    A=np.real(A)
    A.sort()
    print(A)
    # ans=[]
    # for j in range(len(A)):
    #     if A[j]<=1e-5 or A[j]<=-1e-5:
    #         ans.append(0)
    #     else:
    #         ans.append(A[j])
    # A=ans
    # a=ans[:7]
    # B=ans[7:16]
    # C=ans[16:25]
    # # formatted= [format(num, '.2e') for num in A]
    # formatted_A = [format(num, '.2e') for num in a]
    # formatted_B = [format(num, '.2e') for num in B]
    # formatted_C = [format(num, '.2e') for num in C]
    # df = df._append(pd.Series([r[i]] + a), ignore_index=True)
    # df1 = df1._append(pd.Series([r[i]] + formatted_A), ignore_index=True)
    # df2 = df2._append(pd.Series([r[i]] + formatted_B), ignore_index=True)
    # df3 = df3._append(pd.Series([r[i]] + formatted_C), ignore_index=True)
    # print(r[i],"=",[format(num, '.2e') for num in A])
# df.columns = ['r'] + ['eigenvalue'+str(i+1) for i in range(len(a))]
# df1.columns = ['r'] + ['eigenvalue'+str(i+1) for i in range(len(formatted_A))]
# df2.columns = ['r'] + ['eigenvalue'+str(i+8) for i in range(len(formatted_B))]
# df3.columns = ['r'] + ['eigenvalue'+str(i+17) for i in range(len(formatted_C))]

# df.to_excel('eigvals.xlsx', index=False)
# df1.to_excel('eigvals1-7.xlsx', index=False)
# df2.to_excel('eigvals8-16.xlsx', index=False)
# df3.to_excel('eigvals16-24.xlsx', index=False)

# latex_code = df.to_latex(index=False)
# print(latex_code)

# df_trans=df.T
# print(df_trans)




