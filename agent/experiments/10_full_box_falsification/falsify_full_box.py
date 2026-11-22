# Falsification over the whole MNIST input box [0,1]^784 for the mnist68 pair (ε = 1): random images, then
# sign-gradient ascent on max_i |N1(x)_i - N2(x)_i|. Result (this container): random 2.51, after ascent 40.1.
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '06_mnist68_weight_reasoning'))
from load import *
(W1, b1), (W2, b2) = nets(); relu = lambda v: np.maximum(v, 0)
def out(W, b, X):
    a = X
    for k in range(4):
        h = a @ W[k].T + b[k]; a = relu(h) if k < 3 else h
    return a
def dmax(X): return np.abs(out(W1, b1, X) - out(W2, b2, X)).max(1)
rng = np.random.default_rng(0)
X = rng.random((20000, 784)); print('uniform random images: max diff', dmax(X).max())
best = X[np.argsort(-dmax(X))[:20]]
for it in range(150):
    f = dmax(best); g = np.zeros_like(best); e = 1e-3
    for j in range(784):
        Y = best.copy(); Y[:, j] = np.clip(Y[:, j] + e, 0, 1); g[:, j] = (dmax(Y) - f) / e
    best = np.clip(best + 0.05 * np.sign(g), 0, 1)
print('after ascent over [0,1]^784: max diff', dmax(best).max(), '(eps = 1.0)')
