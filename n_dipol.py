import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import colors
from scipy.constants import c, e, mu_0

from pycharge import Charge, potentials_and_fields

jax.config.update("jax_enable_x64", True)

# =====================================================
# 1) PARAMETRELER
# =====================================================
N = 20                      # 🔥 İSTEDİĞİN SAYI
amplitude = 5e-11
omega = (0.8 * c) / amplitude
q = e

# Dipoller arası mesafe
d = 2.0e-10

print(f"N = {N}")
print(f"v_max/c = {float(amplitude*omega/c):.3f}")

# =====================================================
# 2) N DİPOL ÜRET
# =====================================================

# faz seçenekleri:
# hepsi aynı faz:
# phases = jnp.zeros(N)

# alternating (subradiance eğilimi):
phases = jnp.pi * (jnp.arange(N) % 2)

charges = []

for i in range(N):
    xi = (i - (N-1)/2.0) * d
    phi = float(phases[i])

    def pos_plus(t, xi=xi, phi=phi):
        return (xi + amplitude*jnp.sin(omega*t + phi), 0.0, 0.0)

    def pos_minus(t, xi=xi, phi=phi):
        return (xi - amplitude*jnp.sin(omega*t + phi), 0.0, 0.0)

    charges.append(Charge(position_fn=pos_plus,  q=+q))
    charges.append(Charge(position_fn=pos_minus, q=-q))

print(f"Total charges = {len(charges)}")

# =====================================================
# 3) GRID + TIME
# =====================================================
grid_res = 150          # N=100 için çok büyütme yoksa RAM gider
grid_extent = 1e-9

x = jnp.linspace(-grid_extent, grid_extent, grid_res)
y = jnp.linspace(-grid_extent, grid_extent, grid_res)
z = jnp.array([0.0])

nframes = 40
T0 = 2.0 * jnp.pi / omega
t = jnp.linspace(0.0, T0, nframes)

X, Y, Z, T = jnp.meshgrid(x, y, z, t, indexing="ij")

# =====================================================
# 4) ALANLAR
# =====================================================
calculate_fields = jax.jit(potentials_and_fields(charges))
res = calculate_fields(X, Y, Z, T)

E = res.electric.squeeze()     # (Nx,Ny,Nt,3)
B = res.magnetic.squeeze()

Emag = jnp.linalg.norm(E, axis=-1)
Bmag = jnp.linalg.norm(B, axis=-1)

S = jnp.cross(E, B, axis=-1) / mu_0
Smag = jnp.linalg.norm(S, axis=-1)

extent = (float(x[0]), float(x[-1]), float(y[0]), float(y[-1]))

# =====================================================
# 5) LOG SCALE (çok önemli)
# =====================================================
def lognorm(data):
    flat = data.reshape(-1)
    pos = flat[flat > 0]
    if pos.size == 0:
        return colors.LogNorm(vmin=1e-30, vmax=1)
    vmin = float(jnp.quantile(pos, 1e-6))
    vmax = float(jnp.quantile(pos, 0.999))
    return colors.LogNorm(vmin=vmin, vmax=vmax)

norm_E = lognorm(Emag)
norm_B = lognorm(Bmag)
norm_S = lognorm(Smag)

# =====================================================
# 6) ANIMATION
# =====================================================
fig, (axE, axB, axS) = plt.subplots(1,3, figsize=(18,5), constrained_layout=True)

imE = axE.imshow(Emag[:,:,0].T, origin="lower", extent=extent, cmap="viridis", norm=norm_E)
plt.colorbar(imE, ax=axE, label="|E|")

imB = axB.imshow(Bmag[:,:,0].T, origin="lower", extent=extent, cmap="viridis", norm=norm_B)
plt.colorbar(imB, ax=axB, label="|B|")

imS = axS.imshow(Smag[:,:,0].T, origin="lower", extent=extent, cmap="inferno", norm=norm_S)
plt.colorbar(imS, ax=axS, label="|S|")

for ax in (axE, axB, axS):
    ax.set_xlabel("x")
    ax.set_ylabel("y")

def update(i):
    imE.set_data(Emag[:,:,i].T)
    imB.set_data(Bmag[:,:,i].T)
    imS.set_data(Smag[:,:,i].T)

    ti = float(t[i]*1e15)
    axE.set_title(f"|E|  t={ti:.2f} fs")
    axB.set_title(f"|B|  t={ti:.2f} fs")
    axS.set_title(f"|S|  t={ti:.2f} fs  (N={N})")
    return imE, imB, imS

ani = FuncAnimation(fig, update, frames=nframes, interval=50)
plt.show()

ani.save("N_dipole_EB_S.gif", writer="pillow", fps=20)
print("Saved: N_dipole_EB_S.gif")