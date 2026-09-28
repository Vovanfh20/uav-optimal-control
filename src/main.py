"""Запуск расчёта оптимальной посадочной траектории и построение графиков."""

import os
import numpy as np
import matplotlib.pyplot as plt
from dynamics import MISSION
from pmp import optimal_controls
from continuation import solve, trajectory, residual

NU = 1e-3
OUT = "results"


def controls_along(t, y, nu):
    u = np.array([optimal_controls(y[:6, i], y[6:, i], nu) for i in range(len(t))])
    return u


def plot_states(t, y):
    labels = ["v, м/с", "theta, рад", "psi, рад", "h, м", "l, м", "z, м"]
    fig, axes = plt.subplots(3, 2, figsize=(11, 9))
    for i, ax in enumerate(axes.flat):
        ax.plot(t, y[i], color="tab:green")
        ax.set_ylabel(labels[i])
        ax.grid(alpha=0.3)
    axes[0, 0].set_title("Фазовые координаты")
    fig.tight_layout()
    fig.savefig(f"{OUT}/states.png", dpi=150)


def plot_controls(t, u):
    labels = ["nX", "nY", "nZ", "gamma, рад"]
    bounds = [MISSION["nX"], MISSION["nY"], MISSION["nZ"], MISSION["gamma"]]
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t, u[:, i], color="tab:blue")
        ax.axhline(bounds[i][0], color="tab:red", ls="--", alpha=0.5)
        ax.axhline(bounds[i][1], color="tab:red", ls="--", alpha=0.5)
        ax.set_ylabel(labels[i])
        ax.grid(alpha=0.3)
    axes[0].set_title("Оптимальные управления (сглаженные)")
    axes[-1].set_xlabel("время, с")
    fig.tight_layout()
    fig.savefig(f"{OUT}/controls.png", dpi=150)


def plot_trajectory_3d(y):
    l, z, h = y[4], y[5], y[3]
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(l, z, h, color="tab:blue")
    ax.scatter(l[0], z[0], h[0], color="tab:green", s=60, label="старт")
    ax.scatter(l[-1], z[-1], h[-1], color="tab:red", s=60, label="финиш")
    ax.set_xlabel("l, м"); ax.set_ylabel("z, м"); ax.set_zlabel("h, м")
    ax.set_title("Траектория полёта")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/trajectory_3d.png", dpi=150)


def plot_convergence(history):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(np.linspace(0, 1, len(history)), history, "o-", color="tab:red")
    ax.set_xlabel("параметр продолжения mu")
    ax.set_ylabel("||Phi||")
    ax.set_title("Сходимость метода продолжения")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(f"{OUT}/convergence.png", dpi=150)


def main():
    os.makedirs(OUT, exist_ok=True)
    print("Метод продолжения по параметру ...")
    p0, history = solve(nu=NU, n_steps=60)
    final = residual(p0, NU) * np.array([10, 0.1, 0.1, 40, 150, 40])
    print("p0 =", np.round(p0, 4))
    print("норма невязки ||Phi|| =", f"{history[-1]:.3e}")
    print("невязка по координатам =", np.round(final, 3))

    t, y = trajectory(p0, NU)
    u = controls_along(t, y, NU)
    plot_states(t, y)
    plot_controls(t, u)
    plot_trajectory_3d(y)
    plot_convergence(history)
    np.savetxt(f"{OUT}/states.csv", np.column_stack([t, y[:6].T]), delimiter=",")
    print(f"Графики и данные сохранены в '{OUT}/'")


if __name__ == "__main__":
    main()
