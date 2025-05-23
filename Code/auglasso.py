#Figure 1 was generated with the following code:
#>>> sample_range, losses1, losses2 = test(1000, 0.4, 10, 10)
#>>> plot_comparison(sample_range, losses1, losses2)

#Figure 2 was generated with the following code:
#>>> sample_range, losses = testbp(1000, 0.4, 1)
#>>> plot_bp(sample_range, losses)

import numpy
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import pandas
import seaborn

tol = 1e-12

def gram_schmidt_extend(A, v):
    v_gs = v
    for a in A:
        v_gs = v_gs - ((v_gs @ a) / (a @ a)) * a
    if v_gs @ v_gs > tol:
        A.append(v_gs / numpy.linalg.norm(v_gs))   

def gram_schmidt(V): # set of vectors
    A = []
    for v in V:
        gram_schmidt_extend(A, v)
    return A

def find_heavy_coords(V, alpha):
    if len(V) == 0:
        return []
    n = len(V[0])
    A = gram_schmidt(V)
    S = []
    for i in range(n):
        sqcorr = 0
        for v in A:
            sqcorr += v[i]**2
        if sqcorr >= alpha**2:
            S.append(i)
    return S

def iterative_peeling(P, t):
    n = P.shape[0]
    K = set()
    for i in range(n):
        if P[i][i] < 1 - 1/(9*t*t):
            K.add(i)
    for itr in range(t):
        IK = find_heavy_coords([P[i] for i in K], 1/(6*t))
        K.update(set(IK))
    return K

def identify_bad_coords(Sigma, threshold, t):
    Lambda, U = numpy.linalg.eig(Sigma)
    print("Eigenvalues",min(Lambda),max(Lambda))
    goodU = U.T[Lambda >= threshold]
    print(goodU.shape)
    P = goodU.T @ goodU
    print(P.shape)
    K = iterative_peeling(P, t)
    return K

def generate_bad_example(n):
    x = numpy.random.multivariate_normal(numpy.zeros((n)), numpy.eye(n))
    x[5] = x[0] + 1e-4 * x[2]
    x[5] = 10000 * (x[2] - x[0]) + 1e-4 * x[5]
    return x

def generate_bad_cov(n):
    Sigma = numpy.zeros((n,n))
    m = 100*n
    for i in range(m):
        x = generate_bad_example(n)
        Sigma += numpy.outer(x,x)
    return Sigma / m

def bp(X, y, unregset = []):
    m,n = X.shape
    LP = gp.Model("gur-lp")
    LP.params.OutputFlag = 0
    reg_vec = numpy.ones((n))
    for i in unregset:
        reg_vec[i] = 0
    wplus = LP.addMVar(shape=n, vtype=GRB.CONTINUOUS, name="wplus", lb=0, ub=numpy.inf)
    wminus = LP.addMVar(shape=n, vtype=GRB.CONTINUOUS, name="wminus", lb=0, ub=numpy.inf)
    LP.setObjective(reg_vec @ wplus + reg_vec @ wminus)
    LP.addConstr(X @ wplus - X @ wminus == y)
    LP.optimize()
    return wplus.x - wminus.x

def aug_bp(Sigma,threshold,t, X, y, K=None):
    if K == None:
        K = identify_bad_coords(Sigma,threshold,t)
    #print("identified",K)
    return bp(X,y,K)

def generate_samples(Sigma, v, m):
    n = Sigma.shape[0]
    X = numpy.random.multivariate_normal(numpy.zeros((n)),Sigma, size=(m))
    print(X.shape)
    y = numpy.zeros((m))
    for i in range(m):
        y[i] = X[i] @ v
    return X,y

def toy_bad_covariance(n,eps,d):
    Sigma = numpy.eye((n))
    for i in range(d):
        Sigma[3*i][3*i+1] = Sigma[3*i+1][3*i] = 1
        Sigma[3*i+1][3*i+1] = 1 + eps**2
        Sigma[3*i+1][3*i+2] = Sigma[3*i+2][3*i+1] = eps
        Sigma[3*i+2][3*i+2] = 1 + eps**2
    return Sigma

def testbp(n,eps,d,trials=10):
    #sample_range = numpy.linspace(20, 500,50)
    sample_range = numpy.linspace(20, 100,10)
    Sigma = toy_bad_covariance(n,eps,d)
    v = numpy.zeros((n))
    v[0] = eps**-2
    v[1] = -eps**-2
    v[2] = eps**-1
    bp_losses = []
    for m in sample_range:
        print(m)
        bpsub = []
        for iter in range(trials):
            X,y = generate_samples(Sigma, v, int(m))
            v_bp = bp(X,y)
            bpsub.append(min(v@Sigma@v, (v-v_bp) @ Sigma @ (v-v_bp)))
        bp_losses.append(bpsub)
        print(bp_losses[-1])
    return [int(m) for m in sample_range], bp_losses
    
def plot_bp(sample_range, bp_losses):
    df = pandas.DataFrame(columns=["m","bp"])
    for i in range(len(bp_losses)):
        for j in range(len(bp_losses[0])):
            df = df._append({"m": sample_range[i], "bp": bp_losses[i][j]}, ignore_index=True)
    ax1 = seaborn.lineplot(data=df, x="m", y="bp", errorbar="sd", label="BP", color="blue")
    plt.xlabel("Number of samples")
    plt.ylabel("Excess risk")
    plt.legend()
    plt.ylim((-0.1, 1.1))
    plt.show()

def test(n,eps,d,t, trials = 10):
    #sample_range = numpy.linspace(20, 500,50)
    sample_range = numpy.linspace(20, 100,10)
    Sigma = toy_bad_covariance(n,eps,d)
    v = numpy.zeros((n))
    v[0] = eps**-2
    v[1] = -eps**-2
    v[2] = eps**-1
    for i in range(t):
        v[n-1-i] = 1.0/t**0.5
    bp_losses = []
    augbp_losses = []
    K = identify_bad_coords(Sigma,0.1,t+3)
    for m in sample_range:
        print(m)
        bpsub = []
        augbpsub = []
        for iter in range(trials):
            X,y = generate_samples(Sigma, v, int(m))
            v_bp = bp(X,y)
            v_aug = aug_bp(Sigma, 0.1, t+3, X, y, K)
            bpsub.append(min(v@Sigma@v, (v-v_bp) @ Sigma @ (v-v_bp)))
            augbpsub.append(min(v@Sigma@v,(v - v_aug) @ Sigma @ (v - v_aug)))
        bp_losses.append(bpsub)
        augbp_losses.append(augbpsub)
        print(bp_losses[-1],augbp_losses[-1])
    return [int(m) for m in sample_range], bp_losses, augbp_losses

def plot_comparison(sample_range, bp_losses, augbp_losses):
    df = pandas.DataFrame(columns=["m","bp","augbp"])
    for i in range(len(bp_losses)):
        for j in range(len(bp_losses[0])):
            df = df._append({"m": sample_range[i], "bp": bp_losses[i][j], "augbp": augbp_losses[i][j]}, ignore_index=True)
    ax1 = seaborn.lineplot(data=df, x="m", y="bp", errorbar="sd", label="BP", color="blue")
    ax2 = seaborn.lineplot(data=df, x="m", y="augbp", errorbar="sd", label="Augmented BP", color="grey")
    plt.xlabel("Number of samples")
    plt.ylabel("Excess risk")
    plt.legend()
    plt.ylim((-0.1, 2.1))
    plt.show()

# sample_range, losses1, losses2 = test(100, 0.4, 10, 10)
# plot_comparison(sample_range, losses1, losses2)

# sample_range, losses = testbp(100, 0.4, 1)
# plot_bp(sample_range, losses)
