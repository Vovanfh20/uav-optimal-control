"""Решение краевой задачи методом продолжения по параметру.

Пристрелка: подбираем начальный сопряжённый вектор p0 так, чтобы траектория
системы [x, p] за время t_star пришла в целевую точку x_f. Невязку
    Phi(p0) = x(t_star; p0) - x_f
обнуляем не сразу, а деформируя её по параметру mu из [0, 1]:
    Phi(p0(mu)) = (1 - mu) * Phi(p0_start),
что на каждом шаге mu решается демпфированным методом Ньютона (метод «малых
деформаций», Аввакумов–Киселёв). Параметр сглаживания nu регуляризует
релейные управления.
"""

import numpy as np
from scipy.integrate import solve_ivp
from dynamics import MISSION
from pmp import full_rhs

# характерные масштабы компонент невязки (v, theta, psi, h, l, z)
SCALE = np.array([10.0, 0.1, 0.1, 40.0, 150.0, 40.0])


def integrate(p0, nu, rtol=1e-7, atol=1e-9, dense=False):
    y0 = np.concatenate([MISSION["x0"], p0])
    return solve_ivp(full_rhs, (0.0, MISSION["t_star"]), y0, args=(nu,),
                     method="RK45", rtol=rtol, atol=atol, max_step=0.5,
                     dense_output=dense)


def residual(p0, nu, rtol=1e-7):
    sol = integrate(p0, nu, rtol=rtol)
    if not sol.success or not np.all(np.isfinite(sol.y[:, -1])):
        return np.full(6, 1e3)
    return (sol.y[:6, -1] - MISSION["xf"]) / SCALE


def _jacobian(p0, nu, F0, rtol=1e-7):
    J = np.zeros((6, 6))
    for j in range(6):
        step = 1e-5 * max(abs(p0[j]), 1e-2)
        pj = p0.copy()
        pj[j] += step
        J[:, j] = (residual(pj, nu, rtol) - F0) / step
    return J


def _newton_step(p, target, nu, max_iter=4):
    for _ in range(max_iter):
        F = residual(p, nu)
        r = F - target
        if np.linalg.norm(r) < 1e-10:
            break
        J = _jacobian(p, nu, F)
        try:
            dp = np.linalg.solve(J, r)
        except np.linalg.LinAlgError:
            dp = np.linalg.lstsq(J, r, rcond=None)[0]
        norm = np.linalg.norm(dp)
        if norm > 5.0:                       # ограничиваем длину шага
            dp *= 5.0 / norm
        alpha, base = 1.0, np.linalg.norm(r)
        for _ in range(8):                   # дробление шага
            if np.linalg.norm(residual(p - alpha * dp, nu) - target) < base:
                break
            alpha *= 0.5
        p = p - alpha * dp
    return p


def solve(p0=None, nu=1e-2, n_steps=60):
    """Метод продолжения по mu. Возвращает p0 и историю нормы невязки."""
    p = np.zeros(6) if p0 is None else np.array(p0, float)
    Phi0 = residual(p, nu)
    history = [np.linalg.norm(Phi0)]
    for mu in np.linspace(0.0, 1.0, n_steps + 1)[1:]:
        p = _newton_step(p, (1.0 - mu) * Phi0, nu)
        history.append(np.linalg.norm(residual(p, nu)))
    return p, history


def trajectory(p0, nu, n_points=300):
    sol = integrate(p0, nu, rtol=1e-9, atol=1e-11, dense=True)
    t = np.linspace(0.0, MISSION["t_star"], n_points)
    return t, sol.sol(t)
