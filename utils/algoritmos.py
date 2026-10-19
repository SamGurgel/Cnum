import numpy as np
from scipy.optimize import approx_fprime


def JN(x, F, eps=1e-8):
    x = np.asarray(x, dtype=float)
    n = x.size
    Jnum = np.zeros((n, n), dtype=float)

    def fi(v):
        return F(v)[i]

    for i in range(n):
        Jnum[i, :] = approx_fprime(x, fi, epsilon=eps)

    return Jnum


def G(x, F, J):
    return x - np.linalg.inv(J(x)) @ F(x)


def GN(x, F):
    return x - np.linalg.inv(JN(x, F)) @ F(x)


def newton_raphson(F, x0, tol=1e-8, max_iter=100, J=None, eps=1e-8):
    """Resolve sistemas F(x)=0 usando o método de Newton-Raphson.

    Parâmetros
    ----------
    F : callable
        Função vetorial que recebe um vetor x e retorna F(x).
    x0 : array_like
        Aproximação inicial.
    tol : float, optional
        Tolerância para o critério de parada.
    max_iter : int, optional
        Número máximo de iterações.
    J : callable, optional
        Matriz Jacobiana de F. Se não for informada, usa aproximação numérica.
    eps : float, optional
        Passo finito para a aproximação numérica do Jacobiano.
    """
    x = np.asarray(x0, dtype=float).copy()

    if J is None:
        J = lambda v: JN(v, F, eps=eps)

    for _ in range(max_iter):
        Fx = np.asarray(F(x), dtype=float)

        if np.linalg.norm(Fx, ord=np.inf) < tol:
            return x

        Jx = np.asarray(J(x), dtype=float)

        try:
            step = np.linalg.solve(Jx, Fx)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(Jx) @ Fx

        x_new = x - step

        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new

        x = x_new

    return x


def fixed_point(a, g, TOL=1e-8, iter=1000):
    x = g(a)
    i = 1
    while np.linalg.norm(x - a) > TOL and i < iter:
        a = x
        x = g(a)
        i += 1
    return x
