"""
Split operator in 2 d
"""
import numpy as np
import matplotlib.pyplot as plt


def potential(q):
    """
    potential energy at q

    Parameters
    ----------
    q : array, float, shape (..., 2)

    Returns
    -------
    V : array, float, shape `q.shape[:-1]`
        Potential values
    """
    q_0 = np.array([4.0, 0.5])
    sigma = 1.0
    arg = np.sum(((q - q_0) / sigma) ** 2, axis=-1)
    V_0 = 1.0
    return V_0 * np.exp(-arg)


def kinetic(p):
    """
    kinetic energy
    """
    return np.sum(p ** 2, axis=-1) / 2


def coherent_state(grid, q_0, p_0, hbar, sigma=1.0):
    """
    returns a coherent state
    """
    a = 2 * sigma ** 2
    C = np.sqrt(2 / (np.pi * a))
    return C * np.exp(
        -np.sum((grid - q_0) ** 2, axis=-1) / a -
        1j * grid @ p_0 / hbar)


x = np.linspace(-3, 7, 70)
y = np.linspace(-5, 5, 65)
grid = np.array(np.meshgrid(
    x, y, indexing='ij')).transpose(
        (1, 2, 0))

V = potential(grid)

hbar = 0.05
q_0 = np.array([-1.0, 0.5])
p_0 = np.array([1.0, 0.1])
psi_0 = coherent_state(
    grid, q_0, p_0, hbar)


fig = plt.figure()
ax = fig.add_subplot(
    111, title=r"$\psi(q)$",
    xlabel="$x$", ylabel="$y$")

ax.contour(
    grid[..., 0],
    grid[..., 1],
    V,
    cmap=plt.cm.Greys_r)

psi_0_abs2 = np.abs(psi_0) ** 2
ax.imshow(
    psi_0_abs2.T,
    origin="lower",
    interpolation="nearest",
    extent=(x.min(), x.max(),
            y.min(), y.max()),
    cmap=plt.cm.viridis)

plt.show()
