import sys; sys.path.insert(0,'.')
from load import *
from scipy.optimize import linprog
lo,hi=spec(); free=np.where(lo<hi)[0]
(W1,b1),(W2,b2)=nets()
c0=W1[0]@lo+b1[0]; G1=W1[0][:,free]; c02=W2[0]@lo+b2[0]; G2=W2[0][:,free]
relu=lambda v: np.maximum(v,0)
def slab(a,b,n=40,seed=1):
    rng=np.random.default_rng(seed); keep=[]
    for _ in range(n):
        X=rng.random((200000,15)); r=X@G1[44]+c0[44]; keep.append(X[(r>=a)&(r<=b)])
    return np.vstack(keep)
def lp_range(g,c,a,b,extra_A=None,extra_b=None):
    A=np.vstack([G1[44],-G1[44]]); bb=np.array([b-c0[44], -(a-c0[44])])
    if extra_A is not None: A=np.vstack([A,extra_A]); bb=np.r_[bb,extra_b]
    mn=linprog(g,A_ub=A,b_ub=bb,bounds=[(0,1)]*15)
    mx=linprog(-g,A_ub=A,b_ub=bb,bounds=[(0,1)]*15)
    return (mn.fun+c if mn.success else np.inf), (-mx.fun+c if mx.success else -np.inf)
