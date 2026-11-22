import sys, json; import os; sys.path[:0]=[os.path.dirname(os.path.abspath(__file__)), os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','06_mnist68_weight_reasoning')]
from truemax import *
from scipy.optimize import linprog
from numpy.linalg import lstsq, pinv
a,b=float(sys.argv[1]),float(sys.argv[2])
X=slab(a,b,n=60); Dv=diff(X)
M=np.c_[np.ones(len(X)),X]; coef=lstsq(M,Dv,rcond=None)[0]; c=coef[1:]
print('samples',len(X),'fit resid std',(Dv-M@coef).std())
# express c·x via layer-1 pre-activations z1 = G1 x + c0  (min-norm α with α G1 = c)
alpha=c@pinv(G1); print('reconstruction error', abs(alpha@G1-c).max())
s0=alpha@c0          # Σ α z1 = c·x + s0
A=np.vstack([G1[44],-G1[44]]); bb=np.array([b-c0[44], -(a-c0[44])])
smin=linprog(c,A_ub=A,b_ub=bb,bounds=[(0,1)]*15).fun; smax=-linprog(-c,A_ub=A,b_ub=bb,bounds=[(0,1)]*15).fun
print('c·x range over slab', smin, smax)
def max_with(extraA,extrab,restarts=30):
    AA=np.vstack([A,extraA]); BB=np.r_[bb,extrab]; best=-1
    rng=np.random.default_rng(0)
    for t in range(restarts):
        x=linprog(-rng.normal(size=15),A_ub=AA,b_ub=BB,bounds=[(0,1)]*15).x
        if x is None: return None
        for it in range(30):
            f0=diff(x[None])[0]; e=1e-4
            gr=np.array([(diff((x+e*np.eye(15)[i])[None])[0]-f0)/e for i in range(15)])
            tr=0.3*0.85**it; y=linprog(-gr,A_ub=AA,b_ub=BB,bounds=[(max(0,x[i]-tr),min(1,x[i]+tr)) for i in range(15)]).x
            if y is not None and diff(y[None])[0]>f0: x=y
        best=max(best,diff(x[None])[0])
    return best
for q in (0.3,0.5,0.7,0.85):
    tau=smin+q*(smax-smin)
    lo_=max_with(c[None],np.array([tau])); hi_=max_with(-c[None],np.array([-tau]))
    print(f'cut at quantile {q}: c·x ≤ {tau:.3f}: true max {lo_:.4f} | ≥: {hi_:.4f}')
json.dump({'alpha':alpha.tolist(),'s0':float(s0),'smin':float(smin),'smax':float(smax)},open(f'oblique_{a}_{b}.json','w'))
