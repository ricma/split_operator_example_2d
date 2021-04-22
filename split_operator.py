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


def p_grid_from_q_grid(hbar, grid):
    """
    returns the grid points in p space
    """
    q_0 = grid[0, 0]
    delta_q = grid[1, 1] - grid[0, 0]
    Ns = grid.shape[:2]

    # choose p to be able to use FFT
    delta_p = 2 * np.pi * hbar / (Ns * delta_q)
    p_0 = -(Ns * delta_p) / 2

    indices = (grid - q_0) / delta_q
    p_grid = p_0 + indices * delta_p

    return p_0, delta_p, p_grid, indices


def p_representation(hbar, grid, psi_q):
    """
    Calculate the p-representation of ψ(q)
    """
    q_0 = grid[0, 0]
    delta_q = grid[1, 1] - grid[0, 0]

    p_0, delta_p, p_grid, indices = p_grid_from_q_grid(hbar, grid)

    a = np.exp(-1j / hbar * grid @ p_0) * psi_q
    b = np.fft.fft2(a)

    # calculate remaining phase factor
    c = b * np.exp(-1j / hbar * (p_grid - p_0) @ q_0)

    normalization = np.prod(delta_q) / (2 * np.pi * hbar)

    return normalization * c


def q_representation(hbar, grid, psi_p):
    """
    Calculate the q-representation of ψ(p)
    """
    q_0 = grid[0, 0]
    delta_q = grid[1, 1] - grid[0, 0]
    p_0, delta_p, p_grid, indices = p_grid_from_q_grid(hbar, grid)

    a = np.exp(1j / hbar * p_grid @ q_0) * psi_p
    Ns = grid.shape
    b = np.prod(Ns) * np.fft.ifft2(a)

    # calculate remaining phase factor
    c = b * np.exp(1j / hbar * (grid - q_0) @ p_0)

    normalization = np.prod(delta_p) / (2 * np.pi * hbar)

    return normalization * c


def propagate(hbar, grid, psi_q, delta_t):
    """
    apply the split operator
    """
    p_0, delta_p, p_grid, indices = p_grid_from_q_grid(hbar, grid)
    potential_phase = np.exp(- delta_t * 1j / hbar * potential(grid) / 2)
    kinetic_phase = np.exp(- delta_t * 1j / hbar * kinetic(p_grid))

    a = p_representation(hbar, grid, potential_phase * psi_q)
    c = q_representation(hbar, grid, kinetic_phase * a)
    return potential_phase * c


x = np.linspace(-4, 7, 70)
y = np.linspace(-5, 5, 65)
grid = np.array(np.meshgrid(
    x, y, indexing='ij')).transpose(
        (1, 2, 0))
delta_q = grid[1, 1] - grid[0, 0]

V = potential(grid)

delta_t = 0.01
hbar = 0.01
q_0 = np.array([0.0, 0.5])
p_0 = np.array([1.0, 0.1])
psi_0 = coherent_state(
    grid, q_0, p_0, hbar)

# check normalization and expectation value for q₀
norm_psi_0 = np.prod(delta_q) * np.sum(np.abs(psi_0) ** 2)
expectation_q = np.prod(delta_q) * np.sum(
    np.abs(psi_0.flatten()[:, np.newaxis]) ** 2 *
    grid.reshape((-1, 2)), axis=0)

np.testing.assert_array_almost_equal(
    norm_psi_0, 1.0,
    err_msg="Initial state |ψ⟩ not normalized")

np.testing.assert_array_almost_equal(
    expectation_q, q_0,
    err_msg="Coherent state does not have correct ⟨ψ|q|ψ⟩ ≠ q₀")

_, delta_p, p_grid, _ = p_grid_from_q_grid(hbar, grid)
psi_0_p = p_representation(hbar, grid, psi_0)
psi_0_q = p_representation(hbar, grid, psi_0_p)

# np.testing.assert_array_almost_equal(
#     psi_0_q, psi_0,
#     err_msg="Fourier trafo and inverse are not inverse")

psi_1 = propagate(hbar, grid, psi_0, delta_t)

fig = plt.figure()
ax = fig.add_subplot(
    121, title=r"$\psi(q)$",
    xlabel="$x$", ylabel="$y$")

ax_p = fig.add_subplot(
    122, title=r"$\psi(p)$",
    xlabel="$p_x$", ylabel="$p_y$")

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

psi_p_0_abs2 = np.abs(psi_0_p) ** 2
ax_p.imshow(
    psi_p_0_abs2.T,
    origin="lower",
    interpolation="nearest",
    extent=(p_grid[0, 0, 0], p_grid[-1, -1, 0],
            p_grid[0, 0, 1], p_grid[-1, -1, 1]),
    cmap=plt.cm.magma)

plt.show()
