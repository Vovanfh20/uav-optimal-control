"""Модель движения БПЛА и функционал качества.

Фазовый вектор x = [v, theta, psi, h, l, z], управление u = [nX, nY, nZ, gamma].
Уравнения (1)-(6) и функция Ляпунова взяты из постановки задачи (Аввакумов,
Киселёв; принцип максимума Понтрягина для посадочной траектории).
"""

import numpy as np

g = 9.81

UAV = {"m": 18.0}  # Орлан-10, оценочная масса, кг

MISSION = {
    "x0": np.array([40.0, np.deg2rad(-2.0), 0.0, 100.0, 0.0, 0.0]),
    "xf": np.array([30.0, 0.0, 0.0, 40.0, 250.0, 0.0]),
    "t_star": 10.0,
    "lambda0": -1e-3,
    "nX": (-0.5, 0.8),
    "nY": (0.5, 2.0),
    "nZ": (-0.3, 0.3),
    "gamma": (np.deg2rad(-30.0), np.deg2rad(30.0)),
}


def rhs(x, u):
    """Правая часть системы (1)-(6): производные фазового вектора."""
    v, theta, psi, h, l, z = x
    nX, nY, nZ, gamma = u
    ct = np.cos(theta)
    if abs(ct) < 1e-9:
        ct = 1e-9
    return np.array([
        g * (nX - np.sin(theta)),
        g * (nY * np.cos(gamma) - nZ * np.sin(gamma) - np.cos(theta)) / v,
        -g * (nY * np.sin(gamma) - nZ * np.cos(gamma)) / (v * ct),
        v * np.sin(theta),
        v * np.cos(theta) * np.cos(psi),
        -v * np.cos(theta) * np.sin(psi),
    ])


def lyapunov(v, h):
    """Функция Ляпунова V(v, h) из функционала качества (lambda0 < 0)."""
    m, lam = UAV["m"], MISSION["lambda0"]
    v0, h0 = MISSION["x0"][0], MISSION["x0"][3]
    T = 0.5 * m * (v - v0) ** 2
    Pi = m * g * (h - h0)
    return lam * (T ** 2 / 2 + (Pi - T) ** 2 / 2)


def lyapunov_grad(v, h):
    """Частные производные dV/dv, dV/dh."""
    m, lam = UAV["m"], MISSION["lambda0"]
    v0, h0 = MISSION["x0"][0], MISSION["x0"][3]
    T = 0.5 * m * (v - v0) ** 2
    Pi = m * g * (h - h0)
    dT = m * (v - v0)
    dVdv = lam * dT * (2 * T - Pi)
    dVdh = lam * (Pi - T) * m * g
    return dVdv, dVdh
