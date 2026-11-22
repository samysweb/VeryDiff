import sys; sys.path.insert(0,'.')
from slab import *
from scipy.optimize import linprog
def diff(X):
    h1=X@G1.T+c0; a=relu(h1); h2=a@W1[1].T+b1[1]; h3=relu(h2)@W1[2].T+b1[2]
    k1=X@G2.T+c02; k2=relu(k1)@W2[1].T+b2[1]; k3=relu(k2)@W2[2].T+b2[2]
    return relu(h3)[:,3]-relu(k3)[:,3]
def band_max(a,b,restarts=60,seed=0):
    rng=np.random.default_rng(seed)
    A=np.vstack([G1[44],-G1[44]]); bb=np.array([b-c0[44], -(a-c0[44])])
    best=-1; bx=None
    for t in range(restarts):
        # start: LP vertex maximising a random linear objective (or the local gradient)
        g=rng.normal(size=15)
        x=linprog(-g,A_ub=A,b_ub=bb,bounds=[(0,1)]*15).x
        for it in range(40):   # successive linearisation (finite-difference gradient) + LP step in a trust region
            f0=diff(x[None])[0]; e=1e-4
            gr=np.array([(diff((x+e*np.eye(15)[i])[None])[0]-f0)/e for i in range(15)])
            tr=0.3*0.85**it
            bnds=[(max(0,x[i]-tr),min(1,x[i]+tr)) for i in range(15)]
            y=linprog(-gr,A_ub=A,b_ub=bb,bounds=bnds).x
            if diff(y[None])[0]>f0: x=y
        f=diff(x[None])[0]
        if f>best: best=f; bx=x
    return best,bx
if __name__=='__main__':
    for a,b in ((0.85,0.95),(0.95,0.975),(0.975,1.0),(1.0,1.025),(1.025,1.05),(1.05,1.133)):
        m,x=band_max(a,b); print(f'band [{a},{b}]: true max diff_4 ≈ {m:.4f}'); sys.stdout.flush()
