#Code By T.Pranay.
#Date:- 18-11-26

#Importing the numpy module.
import numpy as np

#Defining Gram Schmidt.
def gram_schmidt(A):
    A = A.astype(float)
    n_rows,n_cols = A.shape
    Q = np.zeros((n_rows,n_cols),dtype=float)

    for i in range(n_cols):
          q = A[:,i] - Q[:,:i]@(Q[:,:i].T@A[:,i])
          q /= np.linalg.norm(q)
          Q[:,i] = q

    return Q

def back_substitution(Rmat, cvec):
    m, n = Rmat.shape
    x = cvec.astype(float)
    for i in range(n-1, -1, -1):
        for j in range(n-1, i, -1):
            x[i] -= Rmat[i,j]*x[j]                 
        x[i] /= Rmat[i, i]                 
    return x   

#Define Matrix A (3x3).

A = np.array([[1,1,0],
             [1,0,1],
             [0,1,1]])

#Applying the Gram Schmidt Logic To All the Column Vectors of A.
Q = gram_schmidt(A)

#Computing R = Q^T*A
R = Q.T@A

"""Printing R And Rounding Off All the elements to 10 decimal points 
as there might be the error due to the computation capabilities 
of a computer and hence we will never get R an upper triangular 
due to the errors of the order 10^-16 in this case, hence rounding of 
to 10 places is a very good approximation"""

#Printing R
print(f"The Matrix R is \n{np.round(R,10)}\nand is upper triangular")

#Final Verifications

"""||Q^TQ - I|| = 0 (order of 10^-16), for this to be universal,
I am taking a tall matrix m*n and the q will be m*n and the 
order of QTQ = n*n and QQT is m*m and hence defining I(N) , I(M)"""

mA, nA = A.shape

I_N = np.identity(nA) 
norm1 = np.linalg.norm((Q.T)@Q - I_N)
#This Happens When Q^TQ = I
print(f"\nThe Norm Of the matrix (Q^TQ - I) is {norm1} (approx 0)\nImplying Q^TQ = I and is {np.allclose((Q.T)@Q,I_N)}.")

#Q@R = A Verification
print(f"\nThe Product QR = \n{np.round(Q@R,2)}\n\n= A = \n{A}\n\nThis is {np.allclose(Q @ R, A)} (QR = A)")
norm2 = np.linalg.norm(Q@R - A)

"""The Norm of the matrix QR - A is 0 as all the 
elements is zero implies QR(i,j) = A(i,j)""" 

print(f"\nThe Norm of the matrix (Q*R - A) is {norm2} (approx 0)")


"""Checking if the matrix is upper triangular, Taking only the upper tri 
values of R and forming a matrix with that and comparing with R will yield 
true if and only if R is upper triangular"""

print(f"\nThe Matrix R is \n\n{R}\n\n")
print(f"The Statement that R is an Upper Triangular is {np.allclose(R, np.triu(R))}")

#Coming To Part D, Generating the random Matrix.
np.random.seed(0)
B = np.random.randn(5, 3)
b = np.random.randn(5)

Q = gram_schmidt(B)

"""For Tall matrices inverses of Q wont exist, since Q is also a tall matrix, but that 
of Q^TQ and QQ^T Exists (both being square matrices) hence, QR is B, so QT@Q@R 
is Q^T@B hence R is (Q^T@Q)^-1@Q^T@B implying R is still Q^T@B. The projection 
matrix takes the vector's projection into its subspace. which is hence P^2 is P."""

R = Q.T@B

Pa = (Q.T)@Q
P = Q@(Q.T)
mB,nB = B.shape

print(f"Q^TQ is \n\n{np.round(Pa,10)}\n")
print(f"QQ^t is \n\n{np.round(P,10)}\n")
print(f"The Statement Q^TQ (Pa) is I is {np.allclose(Pa,np.identity(nB))}")
print(f"The Statement QQ^T (P) is I is {np.allclose(P,np.identity(mB))}")

print(f"\nP^2 is \n\n{P@P}\n")
print(f"\nP^T is \n\n{P.T}\n")

print(f"The Statement P^2 is {np.allclose(P@P,P)}\nand the Statement P^T = P is {np.allclose(P.T,P)}\n")

#P is the projection matrix as per theory hence we name Pb as proj_b = p
p = P@b

for i in range(0,mB):
    print(f"The dot product of (b-p) and P{i+1} column vector is {np.round(((b-p).T)@P[:,i],10)}")

if np.linalg.norm(np.round(P@(p-b),10)) == 0:
    print("\nHence (b-p) is perpendicular to the column vectors of P (orthogonal).")

#E-Part of the question, Solving For x
Q = gram_schmidt(A)
R = (Q.T)@A
d = np.array([1,2,3])
c = (Q.T)@d

#Solving 

x_back = back_substitution(R,c)
x_np = np.linalg.solve(A,d)

print(f"\nThe Back substitution solution is \n{x_back}\n\nand the numpy solution is \n{x_np}.\n\n")
print(f"The difference in the solutions is {x_back-x_np}")
