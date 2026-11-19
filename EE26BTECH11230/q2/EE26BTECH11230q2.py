
#Code By T.Pranay.
#Date:- 19-11-26

#Importing the required modules.
import numpy as np
import matplotlib.pyplot as plt

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

#Defining Back Substitution.
def back_substitution(Rmat,cvec):
    m,n = Rmat.shape
    x = cvec.astype(float)

    for i in range(n-1,-1,-1):
        for j in range(n-1,i,-1):
            x[i] -= Rmat[i,j]*x[j]
        x[i] /= Rmat[i,i]

    return x

#Part A:- Generating the Dataset.

np.random.seed(7)
n = 40

x = np.random.uniform(0,10,n)
y = 4 + 2.5*x + 2*np.random.randn(n)

#Plotting the Scatter Plot.
plt.figure()
plt.scatter(x,y,label="Data Points")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Scatter Plot of the Given Dataset")
plt.legend()
plt.grid()
plt.savefig("Points.pdf")
plt.show()

#Constructing the Design Matrix.
X = np.column_stack((np.ones(n),x))

print(f"X = \n\n{X}\n")

print(f"The Shape of the Design Matrix X is {X.shape}")
print(f"The Rank of the Design Matrix X is {np.linalg.matrix_rank(X)}")

#Part B:- Solving Using Normal Equations.

beta = np.linalg.solve(X.T@X,X.T@y)

print(f"\nThe Value of beta_0 is {beta[0]}")
print(f"The Value of beta_1 is {beta[1]}")
print(f"\nThe Actual Value of beta_0 is 4")
print(f"The Actual Value of beta_1 is 2.5")

#Comparing with Numpy Least Squares.
beta_lstsq = np.linalg.lstsq(X,y,rcond=None)[0]
print(f"\nThe Solution Using np.linalg.lstsq is \n\n{beta_lstsq}")
print(f"The Statement that Both Solutions are Equal is {np.allclose(beta,beta_lstsq)}")

#Plotting the Fitted Regression Line.
x_line = np.linspace(np.min(x),np.max(x),200)
y_line = beta[0] + beta[1]*x_line

plt.figure()
plt.scatter(x,y,label="Data Points")
plt.plot(x_line,y_line,label="Fitted Regression Line",color="red")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression Using Least Squares")
plt.legend()
plt.grid()
plt.savefig("Fit-Model.pdf")
plt.show()

#Part C:- Geometry of the Least Squares Fit.

y_cap = X@beta
r = y-y_cap

#Checking Orthogonality of Residuals.
orthogonality = X.T@r
print(f"\nThe Statement X.T@r = 0 is {np.allclose(orthogonality,np.zeros(X.shape[1]))}")
print(f"The Sum of the Residuals is {np.sum(r)} (approx 0)")

#Defining the Projection Matrix.
P = X@np.linalg.inv(X.T@X)@X.T

#Verifying Projection Matrix Properties.
print(f"\nThe Statement P@y = y_cap is {np.allclose(P@y,y_cap)}")
print(f"The Statement P^2 = P is {np.allclose(P@P,P)}")
trace_P = np.trace(P)
print(f"The Trace of the Projection Matrix P is {trace_P}")
print(f"The Statement Trace(P) = 2 is {np.isclose(trace_P,2)}")

#Verifying the Pythagorean Theorem.
norm_y_sq = np.linalg.norm(y)**2
norm_y_cap_sq = np.linalg.norm(y_cap)**2
norm_r_sq = np.linalg.norm(r)**2

print(f"\nThe Statement ||y||^2 = ||y_cap||^2 + ||r||^2 is {np.isclose(norm_y_sq,norm_y_cap_sq+norm_r_sq)}")

#Part D:- Solving Using QR Decomposition.

#Computing Q and R.
Q = gram_schmidt(X)
R = Q.T@X

#Verifying QR Decomposition.
mX,nX = X.shape
I_N = np.identity(nX)

#Solving R@beta = Q.T@y using Back Substitution.
c = Q.T@y
beta_qr = back_substitution(R,c)

print(f"\nThe Solution Using QR Decomposition and Back Substitution is \n\n{beta_qr}")
print(f"The Statement that Both Solutions are Equal is {np.allclose(beta_qr,beta)}")

