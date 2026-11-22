import onnx, numpy as np, re
from onnx import numpy_helper
import os
D=os.environ.get('INST', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'instances', 'mnist68'))
def layers(p):
    m=onnx.load(p); init={t.name:numpy_helper.to_array(t) for t in m.graph.initializer}
    ops=[]
    for n in m.graph.node:
        ops.append((n.op_type,[init.get(i) for i in n.input if i in init], dict((a.name,onnx.helper.get_attribute_value(a)) for a in n.attribute)))
    return ops
def spec():
    lo={};hi={}
    for l in open(D+'/spec.vnnlib'):
        m=re.match(r'\(assert \(<= ([-\d.e]+) X_(\d+)\)\)',l.strip())
        if m: lo[int(m.group(2))]=float(m.group(1)); continue
        m=re.match(r'\(assert \(<= X_(\d+) ([-\d.e]+)\)\)',l.strip())
        if m: hi[int(m.group(1))]=float(m.group(2))
    n=max(lo)+1
    return np.array([lo[i] for i in range(n)]),np.array([hi[i] for i in range(n)])
def nets():
    out=[]
    for p in ('net1','net2'):
        L=layers(D+'/'+p+'.onnx'); Ws=[a[0] for o,a,_ in L if o=='MatMul']; bs=[a[0] for o,a,_ in L if o=='Add']
        out.append((Ws,bs))
    return out
