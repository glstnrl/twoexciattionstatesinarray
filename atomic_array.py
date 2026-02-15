import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import colors
from scipy.constants import c, e, mu_0

from pycharge import Charge, potentials_and_fields

jax.config.update("jax_enable_x64", True)

# =====================================================
# 1) ATOMIC ARRAY PARAMETRELERİ
# =====================================================
N = 20                   
lam = 800e-9
k = 2.0 * jnp.pi / lam

amplitude = 5e-11
omega = (0.8 * c) / amplitude
q = e

d = lam / 10.0          

sep = 2.5e-11

# ---------------------------
# Faz deseni (atomic array modları)

mode = "staggered"  # "inphase", "staggered", "bloch"

if mode == "inphase":
    phases = jnp.zeros((N,))
elif mode == "staggered":
    phases = jnp.pi * (jnp.arange(N) % 2)      # 0,pi,0,pi,...
elif mode == "bloch":
    q_bloch = 0.7 * k                          # seçilebilir
    phases = q_bloch * d * jnp.arange(N)
else:
    raise ValueError("mode must be inphase/staggered/bloch")


charges = []
x0 = -(N - 1) * d / 2.0

for n in range(N):
    xn = x0 + n * d
    phi = float(phases[n])

    # +q ve -q y yönünde +/- sep etrafında titreşsin:
    def pos_plus(t, xn=xn, phi=phi):
        y = +sep + amplitude * jnp.sin(omega * t + phi)
        return (xn, y, 0.0)

    def pos_minus(t, xn=xn, phi=phi):
        y = -sep - amplitude * jnp.sin(omega * t + phi)
        return (xn, y, 0.0)

    charges.append(Charge(position_fn=pos_plus,  q=+q))
    charges.append(Charge(position_fn=pos_minus, q=-q))

print(f"Total charges = {len(charges)} (N atoms = {N})")

# =====================================================
# 3) GRID + TIME (GIF)
# =====================================================
grid_res = 180
grid_extent = 4.0 * float(lam)     # dalga desenini görmek için daha geniş bak

x = jnp.linspace(-grid_extent, grid_extent, grid_res)
y = jnp.linspace(-grid_extent, grid_extent, grid_res)
z = jnp.array([0.0])

nframes = 50
T0 = 2.0 * jnp.pi / omega
t = jnp.linspace(0.0, T0, nframes)

X, Y, Z, T = jnp.meshgrid(x, y, z, t, indexing="ij")

# =====================================================
# 4) FIELDS
# =====================================================
calculate_fields = jax.jit(potentials_and_fields(charges))
res = calculate_fields(X, Y, Z, T)

E = res.electric.squeeze()     # (Nx,Ny,Nt,3)
B = res.magnetic.squeeze()     # (Nx,Ny,Nt,3)

Emag = jnp.linalg.norm(E, axis=-1)
Bmag = jnp.linalg.norm(B, axis=-1)
S = jnp.cross(E, B, axis=-1) / mu_0
Smag = jnp.linalg.norm(S, axis=-1)

extent = (float(x[0]), float(x[-1]), float(y[0]), float(y[-1]))

# =====================================================
# 5) LOG NORM (|E|,|B|,|S|)
# =====================================================
def lognorm(data):
    flat = data.reshape(-1)
    pos = flat[flat > 0]
    if pos.size == 0:
        return colors.LogNorm(vmin=1e-30, vmax=1.0)
    vmin = float(jnp.quantile(pos, 1e-6))
    vmax = float(jnp.quantile(pos, 0.999))
    if vmax <= vmin:
        vmax = vmin * 10.0
    return colors.LogNorm(vmin=vmin, vmax=vmax)

norm_E = lognorm(Emag)
norm_B = lognorm(Bmag)
norm_S = lognorm(Smag)

# =====================================================
# 6) ANIM + GIF
# =====================================================
fig, (axE, axB, axS) = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)

imE = axE.imshow(Emag[:, :, 0].T, origin="lower", extent=extent, cmap="viridis", norm=norm_E, aspect="equal")
plt.colorbar(imE, ax=axE, fraction=0.046, pad=0.04, label="|E|")
axE.set_title("|E|")

imB = axB.imshow(Bmag[:, :, 0].T, origin="lower", extent=extent, cmap="viridis", norm=norm_B, aspect="equal")
plt.colorbar(imB, ax=axB, fraction=0.046, pad=0.04, label="|B|")
axB.set_title("|B|")

imS = axS.imshow(Smag[:, :, 0].T, origin="lower", extent=extent, cmap="inferno", norm=norm_S, aspect="equal")
plt.colorbar(imS, ax=axS, fraction=0.046, pad=0.04, label="|S|")
axS.set_title("|S| (Poynting)")

for ax in (axE, axB, axS):
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

def update(i):
    imE.set_data(Emag[:, :, i].T)
    imB.set_data(Bmag[:, :, i].T)
    imS.set_data(Smag[:, :, i].T)

    ti = float(t[i] * 1e15)
    axE.set_title(f"|E|  t={ti:.2f} fs  mode={mode}")
    axB.set_title(f"|B|  t={ti:.2f} fs  mode={mode}")
    axS.set_title(f"|S|  t={ti:.2f} fs  N={N}  mode={mode}")
    return imE, imB, imS

ani = FuncAnimation(fig, update, frames=nframes, interval=60, blit=False)
plt.show()

ani.save(f"atomic_array_EB_S_{mode}_N{N}.gif", writer="pillow", fps=18)
print("Saved GIF.")