import sys; sys.path.insert(0,'.')
from load import *
lo,hi=spec(); free=np.where(lo<hi)[0]
(W1,b1),(W2,b2)=nets()
P1=[30,37,44,79,94]; P2=[9,36,37,50,85]
relu=lambda v: np.maximum(v,0)
def fwd(W,b,X):
    a=X; hs=[]
    for k in range(4):
        h=a@W[k].T+b[k]; hs.append(h); a=relu(h) if k<3 else h
    return hs
rng=np.random.default_rng(0); N=200000
X=np.tile(lo,(N,1)); X[:,free]=rng.random((N,len(free)))
C=np.tile(lo,(20000,1)); C[:,free]=rng.integers(0,2,(20000,len(free))); X=np.vstack([X,C])
H1=fwd(W1,b1,X); H2=fwd(W2,b2,X); D=H1[3]-H2[3]; r=H1[0][:,44]
