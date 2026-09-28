"""Принцип максимума: сглаженные оптимальные управления и сопряжённая система.

Оптимальные управления имеют релейный характер (sign / sat от переключающих
функций). Для дифференцируемости краевой задачи они сглажены параметром nu:
    SGN(s, nu) = s / sqrt(nu + s^2),
    SAT(s, nu) = (sqrt(nu+(s+1)^2) - sqrt(nu+(s-1)^2)) / 2.
Сопряжённая система pdot = -dH/dx выписана аналитически.
"""

import numpy as np
from dynamics import g, MISSION, rhs, lyapunov_grad


def _sgn(s, nu):
    return s / np.sqrt(nu + s * s)


def _sat(s, nu):
    return 0.5 * (np.sqrt(nu + (s + 1) ** 2) - np.sqrt(nu + (s - 1) ** 2))


def _mid_half(bound):
    lo, hi = bound
    return 0.5 * (hi + lo), 0.5 * (hi - lo)


def optimal_controls(x, p, nu):
    """Сглаженные управления, доставляющие максимум гамильтониану.

    nY, nZ и gamma связаны, поэтому уточняются несколькими итерациями.
    """
    v, theta, psi, h, l, z = x
    pv, ptheta, ppsi, ph, pl, pz = p
    ct = np.cos(theta)
    if abs(ct) < 1e-9:
        ct = 1e-9

    cX, rX = _mid_half(MISSION["nX"])
    cY, rY = _mid_half(MISSION["nY"])
    cZ, rZ = _mid_half(MISSION["nZ"])
    cG, rG = _mid_half(MISSION["gamma"])

    nX = cX + rX * _sgn(pv, nu)
    nY, nZ, gamma = cY, cZ, cG
    for _ in range(4):
        denom = ptheta * nY * ct + ppsi * nZ
        numer = ptheta * nZ * ct + ppsi * nY
        if abs(denom) < 1e-12:
            denom = 1e-12
        s_gamma = -np.arctan2(numer, denom)
        gamma = cG + rG * _sat(s_gamma, nu)

        SY = (g / v) * (ptheta * np.cos(gamma) - ppsi * np.sin(gamma) / ct)
        nY = cY + rY * _sgn(SY, nu)
        SZ = (g / v) * (-ptheta * np.sin(gamma) + ppsi * np.cos(gamma) / ct)
        nZ = cZ + rZ * _sgn(SZ, nu)
    return np.array([nX, nY, nZ, gamma])


def adjoint(x, p, u):
    """Сопряжённая система pdot = -dH/dx (управление фиксировано)."""
    v, theta, psi, h, l, z = x
    pv, ptheta, ppsi, ph, pl, pz = p
    nX, nY, nZ, gamma = u
    st, ct = np.sin(theta), np.cos(theta)
    if abs(ct) < 1e-9:
        ct = 1e-9
    A = nY * np.cos(gamma) - nZ * np.sin(gamma) - ct
    B = nY * np.sin(gamma) - nZ * np.cos(gamma)
    dVdv, dVdh = lyapunov_grad(v, h)

    dHdv = (-ptheta * g * A / v ** 2 + ppsi * g * B / (v ** 2 * ct)
            + ph * st + pl * ct * np.cos(psi) - pz * ct * np.sin(psi) + dVdv)
    dHdtheta = (-pv * g * ct + ptheta * g * st / v - ppsi * g * B * st / (v * ct ** 2)
                + ph * v * ct - pl * v * st * np.cos(psi) + pz * v * st * np.sin(psi))
    dHdpsi = -pl * v * ct * np.sin(psi) - pz * v * ct * np.cos(psi)
    dHdh = dVdh
    return -np.array([dHdv, dHdtheta, dHdpsi, dHdh, 0.0, 0.0])


def full_rhs(t, y, nu):
    """Правая часть расширенной системы [x, p] для интегрирования."""
    x, p = y[:6], y[6:]
    u = optimal_controls(x, p, nu)
    return np.concatenate([rhs(x, u), adjoint(x, p, u)])
