# Triangle-LP relaxation of max diff_4 over {x in [0,1]^15, a <= r(x) <= b}, both networks jointly.
# Pre-activation bounds by sequential LP bound tightening (sound up to LP solver tolerance; offline estimate only).
import sys, time; import os; sys.path[:0]=[os.path.dirname(os.path.abspath(__file__)), os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','06_mnist68_weight_reasoning')]
from slab import *
from scipy.optimize import linprog
import scipy.sparse as sp
DIFF=True
def build(a,b,tighten=True,box=None):
    nets=[(c0,G1,W1,b1),(c02,G2,W2,b2)]
    # variable layout: x(15) | per net: z1(100) a1(100) z2(100) a2(100) z3(10) a3(10)
    off={}; n=15
    for k in range(2):
        for nm,sz in (('z1',100),('a1',100),('z2',100),('a2',100),('z3',10),('a3',10)): off[(k,nm)]=n; n+=sz
    Aeq=[];beq=[];Aub=[];bub=[]
    def row(): return np.zeros(n)
    for k,(c,G,W,bb) in enumerate(nets):
        for j in range(100):
            r_=row(); r_[off[(k,'z1')]+j]=1; r_[:15]=-G[j]; Aeq.append(r_); beq.append(c[j])
            r_=row(); r_[off[(k,'z2')]+j]=1; r_[off[(k,'a1')]:off[(k,'a1')]+100]=-W[1][j]; Aeq.append(r_); beq.append(bb[1][j])
        for j in range(10):
            r_=row(); r_[off[(k,'z3')]+j]=1; r_[off[(k,'a2')]:off[(k,'a2')]+100]=-W[2][j]; Aeq.append(r_); beq.append(bb[2][j])
    # band on r = z1[44] of net 1
    r_=row(); r_[off[(0,'z1')]+44]=1; Aub.append(r_); bub.append(b); Aub.append(-r_); bub.append(-a)
    bounds=(box if box is not None else [(0,1)]*15)+[(None,None)]*(n-15)
    lp=lambda obj,Aub_,bub_: linprog(obj,A_ub=np.array(Aub_),b_ub=np.array(bub_),A_eq=np.array(Aeq),b_eq=np.array(beq),bounds=bounds,method='highs')
    def add_relu(k,zn,an,lo,hi):
        for j in range(len(lo)):
            zi=off[(k,zn)]+j; ai=off[(k,an)]+j
            if hi[j]<=0:
                r_=row(); r_[ai]=1; Aeq.append(r_); beq.append(0.0)
            elif lo[j]>=0:
                r_=row(); r_[ai]=1; r_[zi]=-1; Aeq.append(r_); beq.append(0.0)
            else:
                r_=row(); r_[ai]=-1; Aub.append(r_); bub.append(0.0)               # a >= 0
                r_=row(); r_[ai]=-1; r_[zi]=1; Aub.append(r_); bub.append(0.0)     # a >= z
                s=hi[j]/(hi[j]-lo[j]); r_=row(); r_[ai]=1; r_[zi]=-s; Aub.append(r_); bub.append(-s*lo[j])  # chord
    def bounds_of(k,zn,cnt):
        lo=np.zeros(cnt); hi=np.zeros(cnt)
        for j in range(cnt):
            o=row(); o[off[(k,zn)]+j]=1
            lo[j]=lp(o,Aub,bub).fun; hi[j]=-lp(-o,Aub,bub).fun
        return lo-1e-7,hi+1e-7
    def add_diff(zn,an,cnt):
        # t = a(N1) - a(N2), d = z(N1) - z(N2):  min(0,d) <= t <= max(0,d)
        for j in range(cnt):
            o=row(); o[off[(0,zn)]+j]=1; o[off[(1,zn)]+j]=-1
            dl=lp(o,Aub,bub).fun-1e-7; du=-lp(-o,Aub,bub).fun+1e-7
            tr=row(); tr[off[(0,an)]+j]=1; tr[off[(1,an)]+j]=-1
            dd=o
            if dl>=0:   # 0 <= t <= d
                Aub.append(-tr); bub.append(0.0); Aub.append(tr-dd); bub.append(0.0)
            elif du<=0: # d <= t <= 0
                Aub.append(tr); bub.append(0.0); Aub.append(dd-tr); bub.append(0.0)
            else:       # t <= max(0,d) <= chord ; t >= min(0,d) >= chord
                s1=du/(du-dl); Aub.append(tr-s1*dd); bub.append(-s1*dl)          # t <= s1 (d - dl)
                s2=-dl/(du-dl); Aub.append(-(tr)+(1-s1)*dd); bub.append((1-s1)*du)  # t >= (1-s1)(d - du)
    t=time.time()
    for zn,an,cnt in (('z1','a1',100),('z2','a2',100),('z3','a3',10)):
        for k in range(2):
            lo,hi=bounds_of(k,zn,cnt); add_relu(k,zn,an,lo,hi)
        if DIFF: add_diff(zn,an,cnt)
    o=row(); o[off[(0,'a3')]+3]=-1; o[off[(1,'a3')]+3]=1     # maximise a3_4(N1) - a3_4(N2)
    res=lp(o,Aub,bub)
    return -res.fun, time.time()-t
if __name__=='__main__':
    for a,b in [tuple(map(float,s.split(','))) for s in sys.argv[1:]]:
        v,tt=build(a,b); print(f'band [{a},{b}]: triangle-LP bound on diff_4 = {v:.4f}  ({tt:.0f}s)'); sys.stdout.flush()
