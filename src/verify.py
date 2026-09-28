"""Проверка корректности реализации принципа максимума.

Два независимых теста в случайных точках фазового пространства:
  1. аналитическая сопряжённая система совпадает с численной (dH/dx);
  2. найденные управления действительно доставляют максимум гамильтониану
     (ни одно допустимое управление не даёт большего H).
"""

import numpy as np
from dynamics import MISSION, rhs, lyapunov
from pmp import optimal_controls, adjoint


def hamiltonian(x, p, u):
    return p.dot(rhs(x, u)) + lyapunov(x[0], x[3])


def numeric_adjoint(x, p, u):
    out = np.zeros(6)
    for i in range(6):
        h = 1e-6 * max(abs(x[i]), 1.0)
        xp, xm = x.copy(), x.copy()
        xp[i] += h
        xm[i] -= h
        out[i] = -(hamiltonian(xp, p, u) - hamiltonian(xm, p, u)) / (2 * h)
    return out


def run(n=300, nu=1e-4, seed=0):
    rng = np.random.default_rng(seed)
    adj_err, opt_gap = 0.0, 0.0
    for _ in range(n):
        x = np.array([30 + 15 * rng.random(), rng.uniform(-0.25, 0.25),
                      rng.uniform(-0.25, 0.25), 40 + 80 * rng.random(),
                      200 * rng.random(), rng.uniform(-30, 30)])
        p = rng.uniform(-2, 2, 6)
        u = optimal_controls(x, p, nu)

        adj_err = max(adj_err, np.max(np.abs(adjoint(x, p, u) - numeric_adjoint(x, p, u))
                                      / (1 + np.abs(numeric_adjoint(x, p, u)))))
        H_opt = hamiltonian(x, p, u)
        for _ in range(20):
            u_rand = np.array([rng.uniform(*MISSION["nX"]), rng.uniform(*MISSION["nY"]),
                               rng.uniform(*MISSION["nZ"]), rng.uniform(*MISSION["gamma"])])
            opt_gap = max(opt_gap, hamiltonian(x, p, u_rand) - H_opt)
    return adj_err, opt_gap


if __name__ == "__main__":
    adj_err, opt_gap = run()
    print(f"1) |аналит. - числ. сопряжённая|, отн.: {adj_err:.2e}")
    print(f"2) max(H(допустимое) - H(оптим.)):    {opt_gap:.2e}  (<= 0 -> максимум найден верно)")
    assert adj_err < 1e-4 and opt_gap <= 1e-9
    print("Проверка пройдена.")
